import re
import sys

import pandas as pd
import torch

sys.path.append('../')

from Trainer import Trainer
from Decoder import Transformer
from tokenizer_utils import Tokenizer
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import h5py
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import nltk


# from nltk.corpus import words

# nltk.download('words')


def get_chunks_from_text(text, chunk_size=512):
    # Split text into chunks of size chunk_size
    # This is useful for very long texts that need to be split into smaller parts
    # for tokenization

    chunks = [text[i:i + chunk_size-12] for i in range(0, len(text), chunk_size-12)]
    for i in range(len(chunks)):
        chunks[i] = '[CLS] ' + chunks[i] + ' [SEP]'
    return chunks


class DataPreparator:
    def __init__(self, tokenizer, max_length=512):

        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.common_english_words = set(["the", "and", "is", "in", "at", "of", "on", "for", "with", "without"])
        with open('../nltk_data/corpora/words/en', 'r') as f:
            words = f.read().splitlines()
        self.common_english_words = set(words)

    def prepare_datasets(self, text_files_path, train_csv_path, test_csv_path, output_dir):
        """
        Prepares and saves tokenized datasets into separate HDF5 files for training, validation, and testing.
        Args:
            text_files_path (str): Directory containing text files for training.
            train_csv_path (str): Path to the CSV file for training.
            test_csv_path (str): Path to the CSV file to be split for validation and testing.
            output_dir (str): Directory where the HDF5 files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Tokenize and save training data
        train_output_path = os.path.join(output_dir, "train_dataset.hdf5")
        self._tokenize_and_save([text_files_path, train_csv_path], train_output_path, dataset_type="train")

        # Tokenize and split test data for validation and testing
        test_df = pd.read_csv(test_csv_path)
        # Splitting the DataFrame for validation and test sets
        val_df = test_df.sample(frac=0.5, random_state=42)
        test_df.drop(val_df.index, inplace=True)

        val_output_path = os.path.join(output_dir, "val_dataset.hdf5")
        test_output_path = os.path.join(output_dir, "test_dataset.hdf5")
        self._tokenize_and_save_df(val_df, val_output_path, dataset_type="validation")
        self._tokenize_and_save_df(test_df, test_output_path, dataset_type="test")

    def is_english_word(self, word):
        # Basic check to see if a word is an English word.
        # This could be a simple check against a set of common English words.
        # For a more comprehensive solution, consider using a dictionary or an NLP library.
        return word.lower() in self.common_english_words

    def preprocess_text(self, text):
        # Ensure text is a string
        text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters like #, @
        text = re.sub(r'[@#]', '', text)
        # Remove English words by splitting the text and filtering
        words = text.split()
        filtered_words = [word for word in words if not self.is_english_word(word)]
        text = ' '.join(filtered_words)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize_and_save(self, file_paths, output_path, dataset_type):
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset("input_ids", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8',
                              compression="gzip")
            hf.create_dataset("attention_mask", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8',
                              compression="gzip")

            total_entries = 0

            for file_path in file_paths:
                if os.path.isdir(file_path):  # Directory of text files
                    for text_file in os.listdir(file_path):
                        full_path = os.path.join(file_path, text_file)
                        self._process_file(full_path, hf)
                else:  # Single CSV file
                    df = pd.read_csv(file_path)
                    self._tokenize_and_save_df(df, output_path, dataset_type)

            print(f"Total entries processed and saved for {dataset_type}: {total_entries}")

    def _tokenize_and_save_df(self, df, output_path, dataset_type):
        with h5py.File(output_path, 'a') as hf:  # Ensure appending mode
            for _, row in tqdm(df.iterrows(), desc=f"Tokenizing {dataset_type}"):
                text = row['content']  # Assuming 'content' column contains text to tokenize
                text = self.preprocess_text(text)
                self._tokenize_and_append(text, hf)

    def _tokenize_and_append(self, text, hf):
        try:
            tokenizer = Tokenizer(tokenizer=self.tokenizer)
            text = self.preprocess_text(text)
            text_chunks = get_chunks_from_text(text, chunk_size=self.max_length)
            for chunk in text_chunks:
                input_ids, attention_mask = tokenizer.handel_encode(chunk)
                # Check if datasets exist; if not, create them
                if "input_ids" not in hf:
                    hf.create_dataset("input_ids", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8',
                                      compression="gzip")
                if "attention_mask" not in hf:
                    hf.create_dataset("attention_mask", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8',
                                      compression="gzip")

                current_size = hf["input_ids"].shape[0]
                hf["input_ids"].resize((current_size + 1, self.max_length))
                hf["attention_mask"].resize((current_size + 1, self.max_length))
                hf["input_ids"][current_size, :] = input_ids[:self.max_length]
                hf["attention_mask"][current_size, :] = attention_mask[:self.max_length]

        except Exception as e:
            print(f"Error tokenizing row: {e}")

    def _process_file(self, file_path, hf):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    self._tokenize_and_append(line.strip(), hf)


class TimeMasking(torch.nn.Module):
    def __init__(self, max_time_mask=50):
        super().__init__()
        self.max_time_mask = max_time_mask

    def forward(self, x):
        time_mask = torch.randint(0, self.max_time_mask, (1,)).item()
        if time_mask > 0:
            x_len = x.size(0)
            if time_mask >= x_len:
                time_mask = x_len - 1
            mask_start = torch.randint(0, max(1, x_len - time_mask), (1,)).item()
            x[mask_start:mask_start + time_mask] = 0
        return x


class FrequencyMasking(torch.nn.Module):
    def __init__(self, max_mask=10):
        super().__init__()
        self.max_mask = max_mask

    def forward(self, x):
        mask = torch.randint(0, self.max_mask, (1,)).item()
        if mask > 0:
            x_len = x.size(0)
            if mask >= x_len:
                mask = x_len - 1
            mask_start = torch.randint(0, max(1, x_len - mask), (1,)).item()
            x[mask_start:mask_start + mask] = 0
        return x


class PretrainDataset(Dataset):
    def __init__(self, hdf5_file_path, max_time_mask=50, max_freq_mask=10, is_train=True):
        """
        Initializes the dataset from an HDF5 file containing tokenized data.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.
        """
        self.hdf5_file_path = hdf5_file_path
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            self.length = hf['input_ids'].shape[0]

        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.is_train = is_train

        self.augmentations = torch.nn.Sequential(
            TimeMasking(),  # Time masking augmentation
            FrequencyMasking(),  # Frequency masking augmentation
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            input_ids = torch.tensor(hf['input_ids'][idx], dtype=torch.long)

            attention_mask = torch.tensor(hf['attention_mask'][idx], dtype=torch.long)

        # input_ids = input_ids.unsqueeze(-1)  # Add a singleton dimension at the end
        # attention_mask = attention_mask.unsqueeze(-1)  # Add a singleton dimension at the end

        if self.is_train:
            input_ids, attention_mask = self.augmentations(input_ids), self.augmentations(attention_mask)
            # Shift the input_ids one step to the right to create the labels tensor

        # # Remove the singleton dimension after applying the augmentations
        # input_ids = input_ids.squeeze(-1)
        # attention_mask = attention_mask.squeeze(-1)
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        labels = torch.cat([input_ids[1:], torch.tensor([-100])])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch):
    """
    Collates batch data, dynamically padding to the longest sequence in the batch.
    It also prepares 'labels' tensor, which matches 'input_ids' but with padding tokens set to -100.

    Args:
        batch: A list of dictionaries with 'input_ids', 'attention_mask', and 'labels'.

    Returns:
        A dictionary with batched 'input_ids', 'attention_mask', and 'labels', all padded to the same length.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded.to(device),
        'attention_mask': attention_masks_padded.to(device),
        'labels': labels_padded.to(device)
    }


def main():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=512)
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your raw data
    text_files_path = '../KinyaStory/Kinyarwanda_Data/Kinyarwanda_Data'
    train_csv_path = '../KinyaStory/kinyarwanda news/train.csv'
    test_csv_path = '../KinyaStory/kinyarwanda news/test.csv'

    # Output directory for processed data
    output_dir = './pretrain_tokenized_data'

    # Initialize and run your data preparation
    # data_preparator = DataPreparator(tokenizer=tokenizer, max_length=max_length)
    # data_preparator.prepare_datasets(text_files_path, train_csv_path, test_csv_path, output_dir)

    # Assuming the above method saves three HDF5 files: train_dataset.hdf5, val_dataset.hdf5, test_dataset.hdf5

    # Initialize DataLoader for each dataset
    train_dataset = PretrainDataset(os.path.join(output_dir, "train_dataset.hdf5"), is_train=False)
    val_dataset = PretrainDataset(os.path.join(output_dir, "val_dataset.hdf5"), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Creating a subset of the data for testing if train and validation works
    train_subset = torch.utils.data.Subset(train_dataset, torch.arange(0, 100))
    val_subset = torch.utils.data.Subset(val_dataset, torch.arange(0, 100))
    train_subset_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_subset_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model path where checkpoints will be saved
    model_path = 'models'

    # Initialize the training process

    # vocab_size, d_model, num_heads, num_layers, dropout=0.1
    model_config = {
        'vocab_size': len(tokenizer.get_vocab()),
        'd_model': 12288 // 8,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1
    }
    model = Transformer(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters.")
    model_config['num_params'] = num_params
    train_instance = Trainer(model, optimizer, criterion, model_path, model_config)

    # Start training
    # train_instance.train(train_loader, val_loader, epochs=50)
    train_instance.train(train_loader, val_loader, epochs=50)

    # # Save the final model
    train_instance.save_model()


if __name__ == "__main__":
    main()
