import numpy as np
import random
import json
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Tokenization function
def tokenize(sentence):
    return word_tokenize(sentence.lower())

# Stemming function
def stem(word):
    return stemmer.stem(word)

# Bag of words function
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence if w not in stop_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return torch.tensor(bag, dtype=torch.float32)

# Load intents from JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Process intents data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Preprocess words
ignore_chars = ['?', '.', '!', ',', "'s", "'m"]
all_words = [stem(w) for w in all_words if w not in stop_words and w not in ignore_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Training Data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

# Model Definition
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.l2(x)))
        x = self.dropout(x)
        x = self.l3(x)
        return x

# Dataset Class
class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train
        self.n_samples = len(X_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Training Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)
batch_size = 8
num_epochs = 1000
learning_rate = 0.001

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

# Early Stopping Variables
best_loss = float('inf')
patience = 10
trigger_times = 0

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(words)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Early Stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"Final Loss: {avg_loss:.4f}")

# Save Model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. File saved to {FILE}")

# Test Model Function
def test_model(sentence):
    model.eval()
    tokenized = tokenize(sentence)
    bow = bag_of_words(tokenized, all_words).to(device)
    output = model(bow.unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    return tag

# Example Test
test_sentence = "Hello, how are you?"
predicted_tag = test_model(test_sentence)
print(f"Predicted Tag: {predicted_tag}")
