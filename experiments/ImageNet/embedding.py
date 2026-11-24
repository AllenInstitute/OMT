#!/usr/bin/env python3
"""
Extract ResNet50 embeddings from ImageNet images stored in S3 and save as h5ad.

Usage:
    python embedding.py [--split train|val] [--max-images N]

Output:
    <model_dir>/imagenet_embeddings_{split}.h5ad   (model_dir set in config.toml)
        adata.X            : (n_images, 2048) float32 ResNet50 embeddings
        adata.obs columns  : synset_id, class_idx, image_path, split
        adata.uns          : model/bucket/class_to_idx metadata
"""

import argparse
import io
import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import boto3
import s3fs
import anndata as ad

# ── Config ──────────────────────────────────────────────────────────────────
# S3 location and local paths come from config.toml (+ config.local.toml).
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, PATHS

AWS_PROFILE = CONFIG["s3"]["aws_profile"]
BUCKET      = CONFIG["s3"]["bucket"]
S3_BASE     = CONFIG["s3"]["prefix"]

MODEL_PATH = PATHS["resnet_weights"]
SAVE_DIR   = PATHS["model_dir"]

BATCH_SIZE  = 256
NUM_WORKERS = 8

# Standard ImageNet normalisation for pretrained ResNet
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ──────────────────────────────────────────────────────────────────
class S3ImageNetDataset(Dataset):
    """Streams ImageNet JPEG images directly from S3 without local caching."""

    def __init__(self, profile: str, bucket: str, prefix: str,
                 transform=None, max_images: int | None = None):
        self.profile   = profile
        self.bucket    = bucket
        self.transform = transform

        print(f"Listing images under s3://{bucket}/{prefix} …", flush=True)
        session   = boto3.Session(profile_name=profile)
        s3_client = session.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")

        files, synsets = [], []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpeg", ".jpg", ".png")):
                    synset = key.split("/")[-2]
                    files.append(key)
                    synsets.append(synset)
                    if max_images and len(files) >= max_images:
                        break
            if max_images and len(files) >= max_images:
                break

        self.files       = files
        self.synsets     = synsets
        classes          = sorted(set(synsets))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        print(f"Found {len(files):,} images across {len(classes):,} classes.",
              flush=True)

    # s3fs is created lazily so each DataLoader worker gets its own connection
    @property
    def fs(self):
        if not hasattr(self, "_fs") or self._fs is None:
            self._fs = s3fs.S3FileSystem(profile=self.profile)
        return self._fs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s3_path = f"{self.bucket}/{self.files[idx]}"
        with self.fs.open(s3_path, "rb") as fh:
            img = Image.open(io.BytesIO(fh.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.synsets[idx], self.files[idx]


# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(loader: DataLoader, model: torch.nn.Module,
                     device: torch.device):
    embeddings, labels, paths = [], [], []
    total = len(loader)
    with torch.no_grad():
        for idx, (images, batch_labels, batch_paths) in enumerate(loader):
            if idx % 50 == 0:
                print(f"  batch {idx:5d}/{total} | images: "
                      f"{idx * loader.batch_size:,}", flush=True)
            feats = model(images.to(device))
            embeddings.append(feats.cpu().numpy())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
    return np.vstack(embeddings), labels, paths


# ── Build & save AnnData ─────────────────────────────────────────────────────
def save_h5ad(embeddings: np.ndarray, labels: list, paths: list,
              dataset: S3ImageNetDataset, split: str, save_path: str):
    adata = ad.AnnData(X=embeddings.astype(np.float32))

    adata.obs["synset_id"]  = labels
    adata.obs["class_idx"]  = [dataset.class_to_idx[l] for l in labels]
    adata.obs["image_path"] = [p.replace(f"{BUCKET}/", "") for p in paths]
    adata.obs["split"]      = split

    # Reset index to simple integers
    adata.obs.index = [str(i) for i in range(len(labels))]

    adata.uns["model"]        = "resnet50"
    adata.uns["model_path"]   = MODEL_PATH
    adata.uns["bucket"]       = BUCKET
    adata.uns["s3_prefix"]    = f"{S3_BASE}/{split}"
    adata.uns["embedding_dim"] = int(embeddings.shape[1])
    adata.uns["n_classes"]    = len(dataset.class_to_idx)
    # Store class mapping as arrays (h5ad-compatible)
    adata.uns["class_synsets"] = list(dataset.class_to_idx.keys())
    adata.uns["class_indices"] = list(dataset.class_to_idx.values())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adata.write_h5ad(save_path)
    print(f"\nSaved → {save_path}", flush=True)
    print(adata)
    return adata


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="ImageNet split to embed")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional cap on number of images (for testing)")
    args = parser.parse_args()

    split     = args.split
    s3_prefix = f"{S3_BASE}/{split}"
    save_path = os.path.join(SAVE_DIR, f"imagenet_embeddings_{split}.h5ad")

    # Prefer the RTX 2080 Ti (cuda:1); fall back to any available GPU, then CPU
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")   # RTX 2080 Ti
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if 'cuda' in str(device) else 'CPU'})", flush=True)
    print(f"Device: {device}", flush=True)

    # Model
    print("Loading ResNet50 …", flush=True)
    model = models.resnet50()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.fc = torch.nn.Identity()   # drop classification head → 2048-d output
    model.eval()
    model.to(device)

    # Dataset & loader
    dataset = S3ImageNetDataset(AWS_PROFILE, BUCKET, s3_prefix,
                                transform=TRANSFORM,
                                max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # Extract
    print("\nExtracting embeddings …", flush=True)
    embeddings, labels, paths = extract_features(loader, model, device)
    print(f"Embeddings shape: {embeddings.shape}", flush=True)

    # Save
    save_h5ad(embeddings, labels, paths, dataset, split, save_path)


if __name__ == "__main__":
    main()
