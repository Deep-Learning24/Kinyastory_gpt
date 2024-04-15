
import sys
sys.path.append('../')

from KinyaStory.pretrain import PretrainDataset,DataPreparator,collate_fn
from Trainer import Trainer
from Decoder import Decoder
from KinyaStory.tokenizer_utils import handel_encode, handel_decode
import os
from torch.utils.data import DataLoader, Dataset

def main():
    
    tokenizer = handel_encode
    max_length = 128  # or another value based on your model's capabilities
    
    # Paths to your raw data
    text_files_path = '../Kinyarwanda_Data/Kinyarwanda_Data'
    train_csv_path = '../kinyarwanda news/train.csv'
    test_csv_path = '../kinyarwanda news/test.csv'
    
    # Output directory for processed data
    output_dir = '../pretrain_tokenized_data'
    
    # Initialize and run your data preparation
    # data_preparator = DataPreparator(tokenizer=tokenizer, max_length=max_length)
    # data_preparator.prepare_datasets(text_files_path, train_csv_path, test_csv_path, output_dir)
    
    # Assuming the above method saves three HDF5 files: train_dataset.hdf5, val_dataset.hdf5, test_dataset.hdf5
    
    # Initialize DataLoader for each dataset
    train_dataset = PretrainDataset(os.path.join(output_dir, "train_dataset.hdf5"))
    val_dataset = PretrainDataset(os.path.join(output_dir, "val_dataset.hdf5"))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Model path where checkpoints will be saved
    model_path = 'models'
    
    # Initialize the training process
    #def __init__(self, model, optimizer, criterion)
    model = Decoder()
    # train_instance = Train(
    #     model_path=model_path,
    #     tokenizer=tokenizer,  # Note: If your tokenizer needs to be used within Pretrain, ensure it's correctly passed and utilized
    #     device='cuda'  # or 'cpu'
    # )
    
    # Start training
    # train_instance.train(train_loader, val_loader, epochs=50)
    
    # # Save the final model
    # train_instance.save_model()

if __name__ == "__main__":
    main()