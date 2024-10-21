# train_hybrid_classifier.py

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
from transformers import AutoTokenizer, AutoModel
from content_analyzers.text_analyzer import TextAnalyzer
from ml_analyzers.ml_text_analyzer import MLTextAnalyzer
from models.hybrid_classifier import HybridClassifier
from sklearn.model_selection import train_test_split

# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
transformer_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
text_analyzer = TextAnalyzer()
ml_text_analyzer = MLTextAnalyzer()

# Collect and preprocess your dataset
# Assume dataset is a list of (text, label) tuples
dataset = [
    # Human-written texts (label: 0)
    ("The quick brown fox jumps over the lazy dog.", 0),
    ("In the midst of chaos, there is also opportunity.", 0),
    ("To be, or not to be, that is the question.", 0),
    ("She sells seashells by the seashore.", 0),
    ("The sun sets in the west, painting the sky with hues of orange and pink.", 0),
    ("A journey of a thousand miles begins with a single step.", 0),
    ("All that glitters is not gold.", 0),
    ("The only thing we have to fear is fear itself.", 0),
    ("Ask not what your country can do for you; ask what you can do for your country.", 0),
    ("I think, therefore I am.", 0),

    # AI-generated texts (label: 1)
    ("Advancements in machine learning have led to significant breakthroughs in artificial intelligence.", 1),
    ("The neural network was trained on a dataset of images to improve its accuracy.", 1),
    ("Deep learning models require large amounts of data to perform effectively.", 1),
    ("Natural language processing enables computers to understand human language.", 1),
    ("The algorithm optimizes the function by adjusting weights through backpropagation.", 1),
    ("Reinforcement learning allows agents to make decisions in complex environments.", 1),
    ("Generative models can create realistic images and sounds.", 1),
    ("Transfer learning leverages knowledge from one domain to improve performance in another.", 1),
    ("Clustering algorithms group similar data points based on defined criteria.", 1),
    ("Support vector machines are supervised learning models used for classification and regression.", 1),
]

# Feature extraction
features_list = []
labels_list = []

for text, label in dataset:
    # Analyze text
    analysis_result = text_analyzer.analyze(text)
    numerical_features = ml_text_analyzer.get_text_ml_features(analysis_result)

    # Generate embedding
    text_embedding = ml_text_analyzer.get_text_embedding(text)

    # Combine features and embedding
    combined_input = ml_text_analyzer.combine_text_embedding_to_features(text_embedding, numerical_features)

    # Add debug prints
    print(f"Text: {text}")
    print(f"Combined Input Shape: {combined_input.shape}")
    print(f"Text Embedding Shape: {text_embedding.shape}")
    print(f"Numerical Features Length: {len(numerical_features)}")

    features_list.append(combined_input)
    labels_list.append(label)

# Prepare data
X = np.array(features_list)
y = np.array(labels_list)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets and loaders
train_dataset = data_utils.TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)  # Ensure labels are float
)
train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = data_utils.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                       torch.tensor(y_val, dtype=torch.float32))
val_loader = data_utils.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
input_size = X_train.shape[1]
model = HybridClassifier(input_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 10
best_val_loss = float('inf')
patience = 2
trigger_times = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        # Save the best model
        torch.save(model.state_dict(), 'models/hybrid_classifier.pt')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping! Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            break
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
