import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from transformers import AutoModel
import matplotlib.pyplot as plt
import os
%matplotlib inline

csv_folder = "/home/shubham4413/GSOC/Similarity/data/"

SIMILARITY_MODEL  = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL)
bert_model = AutoModel.from_pretrained(SIMILARITY_MODEL)

def tokenize_text(text):
    #Tokenizing text and converting to tensors
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return tokens

def detect_column_types(df):
    #Automatically classifying columns as numerical, categorical, or text.
    numerical_cols = []
    categorical_cols = []
    text_cols = []

    for col in df.columns:
        unique_values = df[col].nunique()
        total_values = len(df[col])

        if df[col].dtype in ['int64', 'float64']:  # Numeric Data
            numerical_cols.append(col)

        elif df[col].dtype == 'object':  # Object/String Data
            avg_length = df[col].astype(str).apply(len).mean()

            if unique_values / total_values < 0.2:  # Few unique values → Categorical
                categorical_cols.append(col)
            elif avg_length > 20:  # Long text → Treat as Text Column
                text_cols.append(col)
            else:  # Short text but many unique values → Treat as Categorical
                categorical_cols.append(col)

    return numerical_cols, categorical_cols, text_cols

def encode_data(df, numerical_cols, categorical_cols, text_cols):
    # Normalize numerical columns
  if numerical_cols:
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le 

    # Convert text columns to BERT embeddings
    def extract_bert_embedding(text):
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            output = bert_model(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token

    if text_cols:
        for col in text_cols:
            df[col + "_bert"] = df[col].astype(str).apply(lambda x: extract_bert_embedding(x))
            df.drop(columns=[col], inplace=True)  # Drop original text column after encoding

    return df

rocessed_data = {}

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)
        print(f"Processing: {file}")

        df = pd.read_csv(file_path, low_memory=False)
        numerical_cols, categorical_cols, text_cols = detect_column_types(df)
        print(f"Detected -> Numerical: {numerical_cols}, Categorical: {categorical_cols}, Text: {text_cols}")

        df_processed = encode_data(df, numerical_cols, categorical_cols, text_cols)
        processed_data[file] = df_processed  # Store processed DataFrame

        # Save processed file
        df_processed.to_pickle(os.path.join(csv_folder, file.replace(".csv", "_processed.pkl")))

print("All files processed successfully!")

#Convering the processed data into dataloader
class CustomDataset(Dataset):
    def __init__(self, df, text_column, numerical_columns, tokenizer):
        self.df = df
        self.text_column = text_column
        self.numerical_columns = numerical_columns
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize text
        text_data = self.tokenizer(row[self.text_column], padding="max_length", truncation=True, return_tensors="pt")
        
        # Extract numerical features
        numerical_data = torch.tensor(row[self.numerical_columns].values, dtype=torch.float32)

        return {
            "input_ids": text_data["input_ids"].squeeze(0),
            "attention_mask": text_data["attention_mask"].squeeze(0),
            "numerical_data": numerical_data
        }

dataset = CustomDataset(processed_data, text_column="text_column", numerical_columns=numerical_columns, tokenizer=tokenizer)
batch_size = 32 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SimilarityModel(nn.Module):
    def __init__(self, bert_model_name="distilbert-base-uncased", numerical_input_dim=10):
        super(SimilarityModel, self).__init__()
        
        # Loading pretrained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_output_dim = self.bert.config.hidden_size  #768 for BERT
        
        # MLP for numerical data
        self.mlp = nn.Sequential(
            nn.Linear(numerical_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion layer (BERT + MLP outputs)
        self.fusion = nn.Linear(self.bert_output_dim + 64, 128)
        
        # Output similarity embedding
        self.output_layer = nn.Linear(128, 64)  # Output size for similarity comparison

    def forward(self, text_inputs, numerical_inputs):
        # BERT embedding for text data
        bert_outputs = self.bert(**text_inputs)
        text_embedding = bert_outputs.last_hidden_state[:, 0, :]  # CLS token representation

        # MLP embedding for numerical data
        numerical_embedding = self.mlp(numerical_inputs)

        # Concatenate text and numerical embeddings
        combined_embedding = torch.cat((text_embedding, numerical_embedding), dim=1)

        # Fusion layer
        fused_output = F.relu(self.fusion(combined_embedding))

        # Final embedding for similarity
        final_embedding = self.output_layer(fused_output)
        return final_embedding

# Example usage
model = SimilarityModel(numerical_input_dim=10)
print(model)

def loss_fn(embedding1, embedding2, labels, margin=0.5):
    # Computing contrastive loss based on cosine similarity.
    similarity = F.cosine_similarity(embedding1, embedding2)
    loss = labels * (1 - similarity) + (1 - labels) * torch.clamp(similarity - margin, min=0.0)
    return loss.mean()

#training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = cosine_similarity_loss

for epoch in range(10):
    for batch in dataloader: 
        text_inputs, numerical_inputs, labels = batch

        # Forward pass
        embedding1 = model(text_inputs[0], numerical_inputs[0])
        embedding2 = model(text_inputs[1], numerical_inputs[1])

        # Compute loss
        loss = loss_fn(embedding1, embedding2, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluating the model

text_inputs, numerical_inputs, labels = batch

embedding1 = model(text_inputs[0], numerical_inputs[0])
embedding2 = model(text_inputs[1], numerical_inputs[1])

# 1) Compute Cosine Similarity
cos_sim = F.cosine_similarity(embedding1, embedding1)

# 2) Compute Euclidean Distance
euclidean_dist = torch.dist(embedding1, embedding2, p=2)

# 3) Contrastive loss
magin = 1.0
c_loss = (1 - label) * torch.pow(euclidean_dist, 2) + label * torch.pow(torch.clamp(margin - euclidean_dist, min=0.0), 2)