# File: models/fomo_model.py
# Purpose: Multimodal FoMO detection (text + image) for EA-RTB

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# -----------------------
# Dataset Class
# -----------------------
class FoMODataset(Dataset):
    def __init__(self, csv_file, image_dir, max_len=64):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                  max_length=self.max_len, return_tensors='pt')
        image_path = os.path.join(self.image_dir, os.path.basename(row['image_path']))
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image,
            'url': row['url']
        }

# -----------------------
# FoMO Detection Model
# -----------------------
class FoMOModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_model_name='resnet50', fusion_hidden=512):
        super(FoMOModel, self).__init__()
        # Text Encoder
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Image Encoder
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_dim = resnet.fc.in_features
        
        # Fusion Head
        self.fc = nn.Sequential(
            nn.Linear(self.text_dim + self.image_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output
        
        image_feat = self.image_encoder(image)
        image_feat = image_feat.view(image_feat.size(0), -1)
        
        combined = torch.cat([text_feat, image_feat], dim=1)
        return self.fc(combined)

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    csv_file = "../dataset/metadata_preprocessed.csv"
    image_dir = "../dataset/images_preprocessed/"
    
    dataset = FoMODataset(csv_file, image_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = FoMOModel()
    for batch in loader:
        scores = model(batch['input_ids'], batch['attention_mask'], batch['image'])
        print("FoMO scores:", scores)
        break
