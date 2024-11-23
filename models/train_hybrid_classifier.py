import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from content_analyzers.text_analyzer import TextAnalyzer
from ml_analyzers.ml_text_analyzer import MLTextAnalyzer
from models.hybrid_classifier import HybridClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import logging
import seaborn as sns

# Configure logging
logging.basicConfig(
    filename='models/training_hybrid_classifier.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

def load_dataset(csv_file='models/data/dataset.csv'):
    """
    Loads the dataset from a CSV file using pandas with UTF-8 encoding.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        list of tuples: Each tuple contains (text, label).
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        # Ensure that 'text' and 'label' columns exist
        required_columns = {'text', 'label'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain {required_columns} columns.")
        # Drop any rows with missing values
        df = df.dropna(subset=['text', 'label'])
        # Convert labels to integers if they aren't already
        df['label'] = df['label'].astype(int)
        # Convert pandas DataFrame to list of tuples
        dataset = list(zip(df['text'].tolist(), df['label'].tolist()))
        logging.info(f"Loaded dataset with {len(dataset)} samples.")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise e

def split_dataset(dataset, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and validation sets.

    Parameters:
        dataset (list of tuples): The combined dataset.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_val, y_train, y_val
    """
    try:
        texts, labels = zip(*dataset)
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        logging.info(f"Split dataset into {len(X_train)} training and {len(X_val)} validation samples.")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")
        raise e

def prepare_dataloader(X, y, batch_size=32, shuffle=True, num_workers=4):
    """
    Prepares a PyTorch DataLoader from features and labels, keeping data on CPU.

    Parameters:
        X (list of np.array): Feature vectors.
        y (list of int/float): Labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    try:
        # Convert lists to numpy arrays
        X_np = np.stack(X).astype(np.float32)
        y_np = np.array(y).astype(np.float32)

        # Convert to PyTorch tensors on CPU
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_np)

        dataset = data_utils.TensorDataset(X_tensor, y_tensor)
        loader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Speeds up transfer to GPU
        )
        logging.info(f"Prepared DataLoader with {len(loader)} batches.")
        return loader
    except Exception as e:
        logging.error(f"Error preparing DataLoader: {e}")
        raise e

def extract_features_batch(texts, text_analyzer, ml_text_analyzer, batch_size=32):
    """
    Extracts combined features and embeddings from texts in batches.

    Parameters:
        texts (list of str): List of text samples.
        text_analyzer (TextAnalyzer): Instance of TextAnalyzer.
        ml_text_analyzer (MLTextAnalyzer): Instance of MLTextAnalyzer.
        batch_size (int): Number of texts to process in a batch.

    Returns:
        list of np.array: Combined feature vectors for each text.
    """
    features = []
    num_texts = len(texts)
    logging.info(f"Starting feature extraction for {num_texts} texts with batch size {batch_size}.")

    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Analyze texts in batch using nlp.pipe
            analysis_results = text_analyzer.analyze_batch(batch_texts, batch_size=batch_size)
            numerical_features = [ml_text_analyzer.get_text_ml_features(ar) for ar in analysis_results]

            # Generate embeddings in batch
            embeddings = ml_text_analyzer.get_text_embedding_batch(batch_texts)

            # Combine features and embeddings
            combined_inputs = [
                ml_text_analyzer.combine_text_embedding_to_features(emb, num_feat)
                for emb, num_feat in zip(embeddings, numerical_features)
            ]

            # Verify feature lengths
            if combined_inputs:
                feature_length = len(combined_inputs[0])
                for idx, feat in enumerate(combined_inputs):
                    if len(feat) != feature_length:
                        logging.error(f"Inconsistent feature length at index {i + idx}: {len(feat)} != {feature_length}")
                        combined_inputs[idx] = np.zeros(feature_length, dtype=np.float32)

            features.extend(combined_inputs)
            logging.info(f"Processed batch {i//batch_size +1}: {len(combined_inputs)} features extracted.")
        except Exception as e:
            logging.error(f"Error extracting features for batch starting at index {i}: {e}")
            # Append zero vectors for failed samples
            for _ in batch_texts:
                features.append(np.zeros(ml_text_analyzer.input_size, dtype=np.float32))
    logging.info("Completed feature extraction.")
    return features

def plot_roc_curve(y_true, y_scores, save_path='models/plots/roc_curve.png'):
    """
    Plots the ROC Curve and saves it as an image file.

    Parameters:
        y_true (list): True binary labels.
        y_scores (list): Predicted scores or probabilities.
        save_path (str): Path to save the ROC curve image.
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"ROC curve saved to {save_path}")
        print(f"ROC curve saved to {save_path}")
    except Exception as e:
        logging.error(f"Error plotting ROC curve: {e}")

def plot_confusion_matrix_func(y_true, y_pred, save_path='models/plots/confusion_matrix.png'):
    """
    Plots and saves the confusion matrix.

    Parameters:
        y_true (list): True binary labels.
        y_pred (list): Predicted binary labels.
        save_path (str): Path to save the confusion matrix image.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'AI-Generated'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {save_path}")
        print(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {e}")

def plot_correlation_matrix(csv_file):
    """
    Plots the correlation matrix of features and labels.

    Parameters:
        csv_file (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        plt.figure(figsize=(20, 18))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title('Correlation Matrix')
        plt.savefig('models/plots/correlation_matrix.png')
        plt.close()
        print("Correlation matrix plot saved as correlation_matrix.png")
    except Exception as e:
        print(f"Error plotting correlation matrix: {e}")
        
def save_features_to_csv(texts, labels, features, filename):
    """
    Saves the features along with texts and labels to a CSV file.

    Parameters:
        texts (list of str): Original text samples.
        labels (list of int): Corresponding labels.
        features (list of np.array): Extracted feature vectors.
        filename (str): Output CSV file path.
    """
    try:
        # Convert features to a list of lists
        feature_list = features  # Assuming features is a list of np.array

        # Create feature column names
        num_features = feature_list[0].shape[0]
        feature_names = [f"feature_{i}" for i in range(num_features)]

        # Create a DataFrame for features
        df_features = pd.DataFrame(feature_list, columns=feature_names)

        # Create a DataFrame for texts and labels
        df_labels = pd.DataFrame({
            'text': texts,
            'label': labels
        })

        # Combine the DataFrames
        df_combined = pd.concat([df_labels, df_features], axis=1)

        # Save to CSV
        df_combined.to_csv(filename, index=False, encoding='utf-8')
        logging.info(f"Features saved to {filename}")
        print(f"Features saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving features to {filename}: {e}")
        print(f"Error saving features to {filename}: {e}")
    
def check_perfect_correlation(csv_file, threshold=0.95):
    """
    Checks for features that have perfect or near-perfect correlation with labels.

    Parameters:
        csv_file (str): Path to the CSV file.
        threshold (float): Correlation threshold to identify highly correlated features.

    Returns:
        list: Features with high correlation.
    """
    try:
        df = pd.read_csv(csv_file)
        # Separate features and labels
        features = df.drop(['text', 'label'], axis=1)
        labels = df['label']

        correlations = features.corrwith(labels)
        # Features with correlation > threshold or < -threshold
        high_corr = correlations[abs(correlations) > threshold]
        print(f"Features with correlation greater than {threshold}:")
        print(high_corr)
        return high_corr.index.tolist()
    except Exception as e:
        print(f"Error checking correlations in {csv_file}: {e}")
        return []  

def plot_feature_distributions(csv_file, feature_name):
    """
    Plots the distribution of a feature for both classes.

    Parameters:
        csv_file (str): Path to the CSV file.
        feature_name (str): Name of the feature to plot.
    """
    try:
        df = pd.read_csv(csv_file)
        sns.histplot(data=df, x=feature_name, hue='label', kde=True, stat="density", common_norm=False)
        plt.title(f"Distribution of {feature_name} by Label")
        plt.savefig(f"{feature_name}_distribution.png")
        plt.close()
        print(f"Distributivon plot saved as {feature_name}_distribution.png")
    except Exception as e:
        print(f"Error plotting distribution for {feature_name}: {e}")

def find_unique_features(csv_file):
    """
    Identifies features that are only active (e.g., non-zero) in one class.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: Features unique to each class.
    """
    try:
        df = pd.read_csv(csv_file)
        features = df.drop(['text', 'label'], axis=1)

        unique_features = {'AI-Generated': [], 'Human-Written': []}

        for feature in features.columns:
            # For binary features, check if the feature is exclusively 1 or 0 in a class
            ai_only = df[feature][df['label'] == 1].unique()
            human_only = df[feature][df['label'] == 0].unique()

            if set(ai_only) == {1}:
                unique_features['AI-Generated'].append(feature)
            if set(human_only) == {1}:
                unique_features['Human-Written'].append(feature)

        print("Features unique to AI-Generated:")
        print(unique_features['AI-Generated'])
        print("\nFeatures unique to Human-Written:")
        print(unique_features['Human-Written'])

        return unique_features
    except Exception as e:
        print(f"Error finding unique features in {csv_file}: {e}")
        return {}

def get_top_features(model, num_features=10):
    """
    Retrieves the top features based on model weights.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        num_features (int): Number of top features to retrieve.

    Returns:
        list: Indices of top features.
    """
    try:
        weights = model.fc1.weight.data.cpu().numpy()
        # Assuming a single output neuron, take absolute weights
        abs_weights = np.abs(weights[0])
        top_indices = np.argsort(abs_weights)[-num_features:]
        print(f"Top {num_features} feature indices:", top_indices)
        return top_indices.tolist()
    except Exception as e:
        print(f"Error retrieving top features: {e}")
        return []

# def plot_shap_values(model, X, feature_names, num_samples=100):
#     """
#     Plots SHAP values for model interpretation.

#     Parameters:
#         model (nn.Module): Trained PyTorch model.
#         X (np.array): Feature matrix.
#         feature_names (list): List of feature names.
#         num_samples (int): Number of samples to use for SHAP analysis.
#     """
#     try:
#         # Convert model to scikit-learn interface
#         from torch.utils.data import TensorDataset, DataLoader
#         import torch

#         class SklearnWrapper:
#             def __init__(self, model):
#                 self.model = model
#                 self.model.eval()

#             def predict_proba(self, X):
#                 X_tensor = torch.from_numpy(X).float().to(device)
#                 with torch.no_grad():
#                     outputs = self.model(X_tensor).squeeze()
#                     probs = torch.sigmoid(outputs).cpu().numpy()
#                 return np.vstack([1 - probs, probs]).T

#         sklearn_model = SklearnWrapper(model)

#         # Initialize SHAP explainer
#         explainer = shap.Explainer(sklearn_model.predict_proba, X[:num_samples])

#         # Calculate SHAP values
#         shap_values = explainer(X[:num_samples])

#         # Plot summary
#         shap.summary_plot(shap_values, feature_names=feature_names, show=False)
#         plt.title('SHAP Summary Plot')
#         plt.savefig('shap_summary.png', bbox_inches='tight')
#         plt.close()
#         print("SHAP summary plot saved as shap_summary.png")
#     except Exception as e:
#         print(f"Error plotting SHAP values: {e}")

def remove_highly_correlated_features(csv_file, threshold=0.95):
    """
    Removes features that are highly correlated with the label.

    Parameters:
        csv_file (str): Path to the CSV file.
        threshold (float): Correlation threshold to identify highly correlated features.

    Returns:
        list: Features to remove.
    """
    try:
        df = pd.read_csv(csv_file)
        # Separate features and labels
        features = df.drop(['text', 'label'], axis=1)
        labels = df['label']

        correlations = features.corrwith(labels)
        # Features with correlation > threshold or < -threshold
        high_corr = correlations[abs(correlations) > threshold]
        features_to_remove = high_corr.index.tolist()

        print(f"Features to remove (correlation > {threshold}): {features_to_remove}")
        return features_to_remove
    except Exception as e:
        print(f"Error removing highly correlated features: {e}")
        return []

def drop_features(csv_file, features_to_remove, output_file):
    """
    Drops specified features from the CSV file and saves the result.

    Parameters:
        csv_file (str): Path to the original CSV file.
        features_to_remove (list): List of feature names to remove.
        output_file (str): Path to save the modified CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        df.drop(columns=features_to_remove, inplace=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Dropped features and saved to {output_file}")
        print(f"Dropped features and saved to {output_file}")
    except Exception as e:
        print(f"Error dropping features from {csv_file}: {e}")
                          
def main():
    # # Load and split the dataset
    # try:
    #     dataset = load_dataset('data/dataset.csv')
    #     print(f"Total samples loaded: {len(dataset)}")
    # except Exception as e:
    #     print(f"Failed to load dataset: {e}")
    #     return

    # try:
    #     X_train_texts, X_val_texts, y_train, y_val = split_dataset(dataset)
    #     print(f"Training samples: {len(X_train_texts)}, Validation samples: {len(X_val_texts)}")
    # except Exception as e:
    #     print(f"Failed to split dataset: {e}")
    #     return

    # # Initialize TextAnalyzer and MLTextAnalyzer
    # text_analyzer = TextAnalyzer(corpus=X_train_texts)
    # ml_text_analyzer = MLTextAnalyzer()
    
    # # Extract features with batching
    # try:
    #     print("Extracting training features in batches...")
    #     X_train = extract_features_batch(X_train_texts, text_analyzer=text_analyzer, ml_text_analyzer=ml_text_analyzer, batch_size=64)
    #     print("Extracting validation features in batches...")
    #     X_val = extract_features_batch(X_val_texts, text_analyzer=text_analyzer, ml_text_analyzer=ml_text_analyzer, batch_size=64)
    # except Exception as e:
    #     print(f"Failed during feature extraction: {e}")
    #     return

    # Save features to CSV
    # try:
    #     save_features_to_csv(X_train_texts, y_train, X_train, 'data/train/train_features.csv')
    #     save_features_to_csv(X_val_texts, y_val, X_val, 'data/val/val_features.csv')
    # except Exception as e:
    #     print(f"Failed to save features: {e}")
    #     return
    
    try:
        cleaned_train_df = pd.read_csv('models/data/train/train_features.csv')
        cleaned_val_df = pd.read_csv('models/data/val/val_features.csv')
        
        X_train_cleaned = cleaned_train_df.drop(['text', 'label'], axis=1).values.tolist()
        y_train_cleaned = cleaned_train_df['label'].tolist()
        X_val_cleaned = cleaned_val_df.drop(['text', 'label'], axis=1).values.tolist()
        y_val_cleaned = cleaned_val_df['label'].tolist()
    except Exception as e:
        print(f"Failed to load cleaned datasets: {e}")
        return
    
    # Prepare DataLoaders
    try:
        train_loader = prepare_dataloader(X_train_cleaned, y_train_cleaned, batch_size=32, shuffle=True, num_workers=4)
        val_loader = prepare_dataloader(X_val_cleaned, y_val_cleaned, batch_size=32, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Failed to prepare DataLoaders: {e}")
        return

    # Initialize model
    try:
        input_size = len(X_train_cleaned[0])
        model = HybridClassifier(input_size).to(device)
        logging.info(f"Initialized HybridClassifier with input size: {input_size}")
        print(f"Initialized HybridClassifier with input size: {input_size}")
    except Exception as e:
        print(f"Failed to initialize model with cleaned data: {e}")
        logging.error(f"Failed to initialize model with cleaned data: {e}")
        return
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 10
    best_val_f1 = 0.0
    patience = 3
    trigger_times = 0

    # Lists to store metrics for ROC plotting
    best_val_true = []
    best_val_scores = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Move data to GPU
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_trues = []
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move data to GPU
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                preds = torch.sigmoid(outputs)
                preds = preds.cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(batch_y.cpu().numpy())

        # Convert probabilities to binary predictions
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

        # Calculate metrics
        accuracy = accuracy_score(all_trues, binary_preds)
        precision = precision_score(all_trues, binary_preds)
        recall = recall_score(all_trues, binary_preds)
        f1 = f1_score(all_trues, binary_preds)
        roc_auc = roc_auc_score(all_trues, all_preds)

        avg_val_loss = sum(val_losses) / len(val_losses)

        # Log metrics
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
        )

        # Print metrics
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )
        print(
            f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
        )

        # Early Stopping based on F1 Score
        if f1 > best_val_f1:
            best_val_f1 = f1
            trigger_times = 0
            best_val_true = all_trues.copy()
            best_val_scores = all_preds.copy()
            # Save the best model
            torch.save(model.state_dict(), 'models/hybrid_classifier.pt')
            logging.info(f"Best model saved at epoch {epoch+1} with F1 Score: {best_val_f1:.4f}")
            print(f"Best model saved at epoch {epoch+1} with F1 Score: {best_val_f1:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping! Best Validation F1 Score: {best_val_f1:.4f}')
                logging.info(f'Early stopping at epoch {epoch+1} with Best Validation F1 Score: {best_val_f1:.4f}')
                break

    # After training loop, plot ROC Curve for the best epoch
    if best_val_true and best_val_scores:
        plot_roc_curve(best_val_true, best_val_scores, save_path='models/plots/roc_curve_cleaned.png')
        # Plot Confusion Matrix
        binary_preds_final = [1 if p >= 0.5 else 0 for p in best_val_scores]
        plot_confusion_matrix_func(best_val_true, binary_preds_final, save_path='models/plots/confusion_matrix_cleaned.png')
    else:
        logging.warning("No valid validation metrics to plot ROC curve and Confusion Matrix.")
        print("No valid validation metrics to plot ROC curve and Confusion Matrix.")

    # Optionally, print final best metrics
    print(f"Training completed. Best Validation F1 Score: {best_val_f1:.4f}")
    logging.info(f"Training completed. Best Validation F1 Score: {best_val_f1:.4f}")

# def cross_validate_model(X, y, model_class, input_size, num_folds=5, num_epochs=10, batch_size=32, patience=3):
#     """
#     Performs k-fold cross-validation on the model.

#     Parameters:
#         X (list of np.array): Feature vectors.
#         y (list of int): Labels.
#         model_class (nn.Module): PyTorch model class.
#         input_size (int): Size of the input layer.
#         num_folds (int): Number of cross-validation folds.
#         num_epochs (int): Number of training epochs.
#         batch_size (int): Training batch size.
#         patience (int): Early stopping patience.

#     Returns:
#         None
#     """
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
#     fold = 1
#     for train_index, val_index in skf.split(X, y):
#         print(f"\nStarting Fold {fold}/{num_folds}")
#         logging.info(f"Starting Fold {fold}/{num_folds}")

#         X_train_fold = [X[i] for i in train_index]
#         y_train_fold = [y[i] for i in train_index]
#         X_val_fold = [X[i] for i in val_index]
#         y_val_fold = [y[i] for i in val_index]

#         # Initialize DataLoaders
#         train_loader = prepare_dataloader(X_train_fold, y_train_fold, batch_size=batch_size, shuffle=True, num_workers=4)
#         val_loader = prepare_dataloader(X_val_fold, y_val_fold, batch_size=batch_size, shuffle=False, num_workers=4)

#         # Initialize model
#         model = model_class(input_size).to(device)

#         # Define loss and optimizer
#         criterion = nn.BCEWithLogitsLoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

#         best_val_f1 = 0.0
#         patience_counter = 0

#         for epoch in range(num_epochs):
#             model.train()
#             total_train_loss = 0.0
#             for batch_X, batch_y in train_loader:
#                 batch_X, batch_y = batch_X.to(device), batch_y.to(device)

#                 optimizer.zero_grad()
#                 outputs = model(batch_X).squeeze()
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 total_train_loss += loss.item()
#             avg_train_loss = total_train_loss / len(train_loader)

#             # Validation
#             model.eval()
#             all_preds = []
#             all_trues = []
#             val_losses = []
#             with torch.no_grad():
#                 for batch_X, batch_y in val_loader:
#                     batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                     outputs = model(batch_X).squeeze()
#                     loss = criterion(outputs, batch_y)
#                     val_losses.append(loss.item())
#                     preds = torch.sigmoid(outputs)
#                     preds = preds.cpu().numpy()
#                     all_preds.extend(preds)
#                     all_trues.extend(batch_y.cpu().numpy())

#             # Convert probabilities to binary predictions
#             binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

#             # Calculate metrics
#             accuracy = accuracy_score(all_trues, binary_preds)
#             precision = precision_score(all_trues, binary_preds)
#             recall = recall_score(all_trues, binary_preds)
#             f1 = f1_score(all_trues, binary_preds)
#             roc_auc = roc_auc_score(all_trues, all_preds)

#             avg_val_loss = sum(val_losses) / len(val_losses)

#             # Log metrics
#             logging.info(
#                 f"Fold {fold}, Epoch [{epoch+1}/{num_epochs}], "
#                 f"Training Loss: {avg_train_loss:.4f}, "
#                 f"Validation Loss: {avg_val_loss:.4f}, "
#                 f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
#                 f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
#             )

#             # Print metrics
#             print(
#                 f"Fold {fold}, Epoch [{epoch+1}/{num_epochs}], "
#                 f"Training Loss: {avg_train_loss:.4f}, "
#                 f"Validation Loss: {avg_val_loss:.4f}"
#             )
#             print(
#                 f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
#                 f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
#             )

#             # Early Stopping based on F1 Score
#             if f1 > best_val_f1:
#                 best_val_f1 = f1
#                 patience_counter = 0
#                 # Save the best model for this fold
#                 torch.save(model.state_dict(), f'models/hybrid_classifier_fold{fold}.pt')
#                 logging.info(f"Best model for Fold {fold} saved with F1 Score: {best_val_f1:.4f}")
#                 print(f"Best model for Fold {fold} saved with F1 Score: {best_val_f1:.4f}")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print(f'Early stopping at epoch {epoch+1} for Fold {fold}!')
#                     logging.info(f'Early stopping at epoch {epoch+1} for Fold {fold} with Best Validation F1 Score: {best_val_f1:.4f}')
#                     break

#         fold += 1
        
if __name__ == "__main__":
    main()
