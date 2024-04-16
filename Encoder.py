import json
from transformers  import AutoTokenizer
import torch
import os

import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm
import time

def encode(tokenizer, text):
    encoding = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_attention_mask=True,
    )
    return encoding['input_ids'], encoding['attention_mask']

def decode(tokenizer, token_ids, skip_special_tokens=False):
    if isinstance(token_ids[0], list):
        return [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
    else:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class KinyaTokenizer(object):
    def __init__(self, dataset_path):
        self.tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=512)
        self.dataset_path = dataset_path
        #self.extend_vocab()

    def is_word_new(self, args):
        word, vocab_set = args
        tokens = self.tokenizer.tokenize(word)
        if any(token not in vocab_set for token in tokens):
            return word
        else:
            return None
    
    def get_tokenizer_vocabulary_size(self):
        return len(self.tokenizer.get_vocab())
    
    def extend_vocab(self):
       # Load the dataset
        df = pd.read_csv(self.dataset_path)

        # Collect all unique words in the dataset
        words = set()
        for _, row in df.iterrows():
            story_input = row['story_input']
            story_output = row['story_output']
            words.update(story_input.split())
            words.update(story_output.split())

        # Convert the tokenizer's vocabulary to a set for faster lookup
        vocab_set = set(self.tokenizer.get_vocab().keys())

        # Find out which words are not in the tokenizer's vocabulary
        with Pool() as p:
            new_words = list(tqdm(p.imap(self.is_word_new, [(word, vocab_set) for word in words]), total=len(words)))
        new_words = [word for word in new_words if word is not None]

        # Get the last token ID in the current vocabulary
        last_token_id = max(self.tokenizer.get_vocab().values())

        # Add the new words to the tokenizer's vocabulary with new token IDs
        for word in tqdm(new_words):
            last_token_id += 1
            self.tokenizer.add_tokens([word])
            self.tokenizer.vocab[word] = last_token_id

        # Save the tokenizer
        self.tokenizer.save_pretrained('./kinyatokenizer')



        # Get the tokenizer configuration
        tokenizer_config = self.tokenizer.get_vocab()

        # Save the tokenizer configuration to a JSON file
        import json
        with open('tokenizer_config.json', 'w') as f:
            json.dump(tokenizer_config, f)

    
    def tokenize_dataset(self):
        df = pd.read_csv(self.dataset_path)
        tokenized_data = []
        for _, row in df.iterrows():
            story_input = row['story_input']
            story_output = row['story_output']
            input_encoding = encode(self.tokenizer, story_input)
            output_encoding = encode(self.tokenizer, story_output)
            tokenized_data.append((input_encoding, output_encoding))
        
        # Save the tokenized data
        torch.save(tokenized_data, 'tokenized_data.pt')
        
        return tokenized_data

    
    def print_sample_tokenized_data(self, tokenized_data):
        for tokenized_sequence in tokenized_data:
            decoded_sequence = decode(self.tokenizer,tokenized_sequence)
            print(decoded_sequence)


if __name__ == "__main__":
    KinyaTokenizer = KinyaTokenizer('kinyastory_data/kinyastory.csv')
    tokenized_data = KinyaTokenizer.tokenize_dataset()
    print("Tokenized data saved as tokenized_data.pt")
    KinyaTokenizer.print_sample_tokenized_data(tokenized_data[0])
    print("Sample tokenized data printed")
    

    