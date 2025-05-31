import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model as TeacherGPT2Model
from models.gpt2 import GPT2Model as StudentGPT2Model
from config import GPT2Config
from tqdm import tqdm
import argparse
import random
import numpy as np
import json
import os

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CNNDailyMailDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data from files
        if split == 'train':
            data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('train')]
        else:
            data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('validation')]
            
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    self.examples.append({
                        'article': example['article'],
                        'highlights': example['highlights']
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine article and highlights with special tokens
        text = f"Article: {example['article']}\nSummary: {example['highlights']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def load_models_and_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Teacher model (original pre-trained)
    teacher_model = TeacherGPT2Model.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Teacher는 학습하지 않음

    # Student model (smaller version)
    student_config = GPT2Config(
        vocab_size=50260,  # teacher와 동일한 vocab size
        hidden_size=384,   # 768 -> 384
        num_hidden_layers=6,  # 12 -> 6
        num_attention_heads=6,  # 12 -> 6
        intermediate_size=1536  # 3072 -> 1536
    )
    student_model = StudentGPT2Model(student_config)
    student_model = student_model.to(device)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
    tokenizer.pad_token = tokenizer.eos_token

    return teacher_model, student_model, tokenizer, device

def distillation_loss(student_logits, teacher_logits, temperature):
    """
    Compute the distillation loss (KL divergence between teacher and student logits)
    """
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_softmax = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)

def train(args):
    # 모델과 토크나이저 로드
    teacher_model, student_model, tokenizer, device = load_models_and_tokenizer()
    
    # 데이터셋 로드
    train_dataset = CNNDailyMailDataset(args.data_dir, tokenizer, args.max_length, split='train')
    val_dataset = CNNDailyMailDataset(args.data_dir, tokenizer, args.max_length, split='validation')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{args.epochs}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_hidden_states = teacher_outputs.last_hidden_state
                teacher_logits = teacher_model.wte(teacher_hidden_states)

            # Student forward pass
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_hidden_states = student_outputs['last_hidden_state']
            student_logits = student_model.hidden_state_to_token(student_hidden_states)

            # Compute losses
            distill_loss = distillation_loss(student_logits, teacher_logits, args.temperature)
            
            # Backward pass
            optimizer.zero_grad()
            distill_loss.backward()
            optimizer.step()

            total_loss += distill_loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{args.epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_hidden_states = teacher_outputs.last_hidden_state
                teacher_logits = teacher_model.wte(teacher_hidden_states)

                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_hidden_states = student_outputs['last_hidden_state']
                student_logits = student_model.hidden_state_to_token(student_hidden_states)

                distill_loss = distillation_loss(student_logits, teacher_logits, args.temperature)
                val_loss += distill_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'config': student_config,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss
            }, f"{args.save_dir}/best_distilled_gpt2.pt")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to CNN/DailyMail dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='distilled_model')
    parser.add_argument('--seed', type=int, default=11711)
    return parser.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)
    train(args)

if __name__ == "__main__":
    main()