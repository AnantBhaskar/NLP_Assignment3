import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Define the dataset class
class TextSimilarityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data.iloc[idx]
        sentence1 = str(pair['sentence1']) if 'sentence1' in pair else ""  # Handle missing values
        sentence2 = str(pair['sentence2']) if 'sentence2' in pair else ""  # Handle missing values
        score = pair['score'] if 'score' in pair else 0  # Handle missing values
        
        encoded_pair = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',  # Choose a different truncation strategy
            return_tensors='pt',
            return_overflowing_tokens=True 
        )
        
        input_ids = encoded_pair['input_ids'].squeeze(0)
        attention_mask = encoded_pair['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': score
        }

# Load data
train_data = pd.read_csv('train.csv', sep='\t')
val_data = pd.read_csv('dev.csv', sep='\t')
test_data = pd.read_csv('sample_test.csv', sep='\t')


print(test_data.columns)

# Handle missing values
train_data = train_data.dropna(subset=['sentence1', 'sentence2', 'score']).reset_index(drop=True)
# Drop missing values from validation data
val_data = val_data.dropna(subset=['sentence1', 'sentence2', 'score']).reset_index(drop=True)
# Correct column names with tabs included
test_data = test_data.dropna(subset=['id', 'sentence1', 'sentence2']).reset_index(drop=True)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define max sequence length and batch size
max_length = 128
batch_size = 32

# Datasets and DataLoaders
train_dataset = TextSimilarityDataset(train_data, tokenizer, max_length)
val_dataset = TextSimilarityDataset(val_data, tokenizer, max_length)
test_dataset = TextSimilarityDataset(test_data, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Setup 1A - BERT Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, no_deprecation_warning=True)

loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze(1)
        
        loss = loss_fn(predictions, scores)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation loop
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].float().to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze(1)
            
            loss = loss_fn(predictions, scores)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}')

# Evaluation for Setup 1A - BERT Model
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].float().numpy()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.logits.squeeze(1).cpu().numpy())
        targets.extend(scores)

corr_coef, _ = pearsonr(predictions, targets)
print(f'Pearson Correlation for Setup 1A: {corr_coef}')

# Setup 1B - Sentence-BERT Model (Cosine Similarity)
sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

val_embeddings = sentence_model.encode(val_data['sentence1'].tolist(), convert_to_tensor=True)
similarities = []

for i in range(len(val_data)):
    embedding1 = val_embeddings[i]
    embedding2 = sentence_model.encode([val_data['sentence2'][i]], convert_to_tensor=True)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    similarities.append(similarity)

# Evaluation for Setup 1B - Sentence-BERT Model
corr_coef, _ = pearsonr(similarities, val_data['score'])
print(f'Pearson Correlation for Setup 1B: {corr_coef}')

# Save predictions to CSV
test_embeddings = sentence_model.encode(test_data['sentence1'].tolist(), convert_to_tensor=True)
test_predicted_scores = []

for i in range(len(test_data)):
    embedding1 = test_embeddings[i]
    embedding2 = sentence_model.encode([test_data['sentence2'][i]], convert_to_tensor=True)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    test_predicted_scores.append(similarity)

test_data['predicted_score'] = test_predicted_scores
test_data.to_csv('sample_demo.csv', index=False)

# Plot Losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
