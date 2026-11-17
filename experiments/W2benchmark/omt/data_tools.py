import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import anndata as ad
import pandas as pd
import os
import pdb
import torch
import json


def load_param_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure it's not an empty line
                try:
                    parsed_line = json.loads(line)  # Parse JSON (dict or value)
                except json.JSONDecodeError:
                    parsed_line = line  # Keep as string if it's not valid JSON
                data.append(parsed_line)
    return data