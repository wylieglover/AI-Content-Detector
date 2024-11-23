from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from models.hybrid_classifier import HybridClassifier
import logging

class MLTextAnalyzer:
    def __init__(self):
        # Use pretrained bert tokenizer and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.transformer_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)

        # Load the classifier
        self.embedding_size = 768  # For BERT-base
        self.num_numerical_features = self.calculate_text_num_numerical_features()
        self.input_size = self.embedding_size + self.num_numerical_features
        
        self.classifier = HybridClassifier(self.input_size).to(self.device)
        self.classifier.load_state_dict(torch.load('models/hybrid_classifier.pt', weights_only=True, map_location=self.device))
        self.classifier.eval()
        
        self.reset_metrics()

    def reset_metrics(self):
        self.true_labels = []
        self.predictions = []
        self.probabilities = []

    def store_prediction(self, true_label, prediction, probability):
        self.true_labels.append(true_label)
        self.predictions.append(prediction)
        self.probabilities.append(probability)

    def calculate_text_num_numerical_features(self):
        # Calculate the number of numerical features based on your feature extraction
        num_punct_features = len(self.expected_punct_marks()) 
        num_pos_features = len(self.expected_pos_tags()) 
        num_dep_features = len(self.expected_dep_tags())
        num_scalar_features = len(self.expected_scaler_features())  
        total_features = num_scalar_features + num_punct_features + num_pos_features + num_dep_features

        return total_features
    
    def analyze_text(self, text_data, analyzed_report):
        # Generate transformer embeddings
        text_embedding = self.get_text_embedding_batch(text_data)
        
        # Get features
        numerical_features = self.get_text_ml_features(analyzed_report)
        
        # Combine embeddings and features
        combined_input = np.concatenate([text_embedding, numerical_features], axis=0)
        
        # Make prediction
        probability, label = self.predict_text(combined_input)
        
        return {
            'ml_text_ai_generated_probability': probability,
            'ml_text_ai_generated_label': label
        }
    
    def combine_text_embedding_to_features(self, embedding, numerical_features):
        # Convert numerical features to numpy array
        numerical_features = np.array(numerical_features)
        
        # Concatenate the embedding and numerical features
        combined_input = np.concatenate([embedding, numerical_features], axis=0)
        
        return combined_input

    def get_text_embedding_batch(self, texts):
        # Tokenize the input text
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Shape: (batch_size, hidden_size)
            
            return cls_embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings for batch: {e}")
            return [np.zeros(self.embedding_size, dtype=np.float32) for _ in texts]

    def get_text_ml_features(self, analyzed_report):
        # Extract numerical features from the report
        features = []
        
        # Add scaler features
        scaler_features = self.expected_scaler_features()
        for name in scaler_features:
            features.append(analyzed_report.get(name, 0.0))
        
        # Add punctuation features
        punctuations = self.expected_punct_marks()
        for p in punctuations:
            features.append(analyzed_report.get(f"punct_{p}", 0))
        
        # Add POS tag distributions
        pos_tags = self.expected_pos_tags()
        for pos in pos_tags:
            features.append(analyzed_report.get(f"pos_{pos}", 0))
        
        # Add dependency parsing features
        dep_tags = self.expected_dep_tags()
        for dep in dep_tags:
            features.append(analyzed_report.get(f"dep_{dep}", 0))
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        # Replace inf values
        features = np.where(np.isinf(features), 0.0, features)
        return features
    
    def expected_scaler_features(self):
        return [
            "word_count", "unique_word_count", "sentence_count", "avg_word_length",
            "avg_sentence_length", "sentence_length_std", "word_length_std", "lexical_diversity",
            "stopword_ratio", "hapax_legomena_ratio", "avg_commas_per_sentence",
            "question_ratio", "sentiment_score", "total_punctuations", "punctuation_variety",
            "type_token_ratio", "honore_statistic"
        ]
        
    def expected_punct_marks(self):
        return [
            '.', ',', ';', ':', '!', '?', '-', '"', "'", '(', ')'
        ]
    def expected_pos_tags(self):
        return [
            'PRON', 'AUX', 'DET', 'ADJ', 'NOUN', 'ADP', 'PROPN', 'PUNCT',
            'SCONJ', 'VERB', 'PART', 'CCONJ', 'ADV'
        ]
    
    def expected_dep_tags(self):
        return [
            'nsubj', 'ROOT', 'det', 'amod', 'compound', 'attr', 'prep',
            'pobj', 'punct', 'advmod', 'aux', 'relcl', 'xcomp', 'conj',
            'discourse', 'expl', 'fixed', 'flat', 'iobj',
            'cc', 'nmod', 'prt', 'dobj', 'advcl', 'poss', 'acomp',
            'pcomp', 'npadvmod', 'acl'
        ]
    
    def predict_text(self, combined_input):
        # Convert to torch tensor
        input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0) 
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.classifier(input_tensor)
        
        probability = torch.sigmoid(output).item()
        label = 'AI-Generated' if probability > 0.5 else 'Human-Written'
        
        return probability, label
