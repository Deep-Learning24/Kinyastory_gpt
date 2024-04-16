from statistics import mean
from tqdm import tqdm
import sys
sys.path.append('../')
from torch.cuda.amp import autocast
import torch
import wandb
import torch.nn as nn
from tokenizer_utils import Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import os
from transformers  import AutoTokenizer

from nltk.translate.bleu_score import SmoothingFunction

class Trainer:
    def __init__(self, model, optimizer, criterion,model_path,config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_path = model_path
        self.config = config
        wandb.login(key="")#API Key is in your wandb account, under settings (wandb.ai/settings)
        
        wandb.init(
            project="project-ablations", 
            config=self.config,
            name = "kinya-gpt-pretrain", ## Wandb creates random run names if you skip this field
            reinit = True, ### Allows reinitalizing runs when you re-run this cell
            id ="kinya-gpt-pretrain", ### Insert specific run id here if you want to resume a previous run
            #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
            )
        
        self.load_model()

    def train(self, train_loader,val_loader, epochs):
        self.model.train()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=1024)
        tokenizer_instance = Tokenizer(tokenizer)
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
    
                with autocast():
                    logits,loss = self.model(**batch)
    
                total_loss += loss.item()
    
                # Calculate the number of correct predictions
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == batch['labels']).sum().item()
                total_correct += correct
    
                self.optimizer.zero_grad()
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
    
                self.optimizer.step()
    
                progress_bar.set_postfix({'Training Loss': f'{loss.item():.4f}', 'Training Accuracy': f'{correct / len(batch["labels"]):.4f}'})
    
            avg_train_loss = total_loss / len(train_loader)
            avg_train_accuracy = total_correct / len(train_loader.dataset)
            eval_loss, avg_bleu, avg_rouge, avg_perplexity = self.evaluate(val_loader,tokenizer_instance)
            results={"avg_train_loss": avg_train_loss, "avg_train_accuracy": avg_train_accuracy, "eval_loss": eval_loss, "avg_bleu": avg_bleu, "avg_rouge": avg_rouge, "avg_perplexity": avg_perplexity,"epoch": epoch+1}
            print(results)
            wandb.log(results)
    
    def evaluate(self, loader,tokenizer_instance):
        self.model.eval()
        total_loss = 0
        bleu_scores = []
        rouge_scores = []
        perplexities = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # get the labels which are not equal to -100
               
                with autocast():
                    logits= self.model(input_ids, attention_mask)
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                #labels = labels[labels != -100]
                predicted_ids = torch.argmax(logits, dim=-1)
                bleu_score = self.calculate_bleu_score(labels, predicted_ids,tokenizer_instance)
                rouge_score = self.calculate_rouge_score(labels, predicted_ids,tokenizer_instance)
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)
                total_loss += loss.item()
        avg_bleu = mean(bleu_scores)
        avg_rouge = mean(rouge_scores)
        avg_loss = total_loss / len(loader)
        return avg_loss, avg_bleu, avg_rouge, mean(perplexities)

    
    def calculate_bleu_score(self, labels, predicted_ids,tokenizer_instance):
        bleu_scores = []
        for i in range(len(labels)):
            # Exclude -100 values from labels before decoding
            label = labels[i]
            proper_label = []
                # Decode label tensor
            for token in label:
                if token>0:
                    proper_label.append(token)
                        
            decoded_label = tokenizer_instance.handel_decode(proper_label)
            predicted = tokenizer_instance.handel_decode(predicted_ids[i])
            bleu_score = sentence_bleu(decoded_label, predicted, smoothing_function=self.smoothie)
            bleu_scores.append(bleu_score)
        return mean(bleu_scores)
    
    def calculate_rouge_score(self, labels, predicted_ids,tokenizer_instance):
        rouge_scores = []
    
        for label, predicted_id in zip(labels, predicted_ids):
            # Filter out -100 values and ensure token IDs are positive
            proper_label = [token for token in label if token > 0]
            
            # Decode both label and prediction
            decoded_label = tokenizer_instance.handel_decode(proper_label)
            predicted = tokenizer_instance.handel_decode(predicted_id)
            
            # Check for empty strings to avoid 'Hypothesis is empty' error
            if not decoded_label.strip() or not predicted.strip():
                print("Skipping empty prediction or reference.")
                continue
            
            try:
                rouge_score = self.rouge.get_scores(predicted, decoded_label, avg=True)
                rouge_scores.append(rouge_score['rouge-l']['f'])
            except Exception as e:
                print(f"Error calculating ROUGE score: {e}")
    
        # Compute the average ROUGE-L F1 score if rouge_scores is not empty, else return 0
        return mean(rouge_scores) if rouge_scores else 0

    def save_model(self, filename="best_gpt2_model.pt"):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, filename))
        print(f'Model saved to {os.path.join(self.model_path, filename)}')

    def load_model(self, filename="best_gpt2_model.pt"):
        model_file = os.path.join(self.model_path, filename)
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file))
            
            print(f'Model loaded from {model_file}')
        else:
            print("Model file not found.")
    
        