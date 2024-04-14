
import json
from transformers  import AutoTokenizer
import torch
import os

import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm
import time

def get_tokenizer_vocabulary_size(tokenizer):
        return len(tokenizer.get_vocab())


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=2048)
    print("Vocabulary size: ", get_tokenizer_vocabulary_size(tokenizer))