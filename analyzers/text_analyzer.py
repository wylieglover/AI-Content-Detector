from typing import Dict, List
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from scripts.analyze import ContentAnalyzer

class TextAnalyzer(ContentAnalyzer):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def analyze(self, text):
        features = {}
        
        # Basic statistics
        features.update(self._compute_basic_stats(text))
        
        # Lexical features
        features.update(self._compute_lexical_features(text))
        
        # Syntactic features
        features.update(self._compute_syntactic_features(text))
        
        # Semantic features
        features.update(self._compute_semantic_features(text))
        
        return features
    
    def _compute_basic_stats(self, text):
        tokens = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        return {
            "word_count": len(tokens),
            "unique_word_count": len(set(tokens)),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in tokens]),
            "avg_sentence_length": np.mean([len(word_tokenize(sent)) for sent in sentences])
        }
    
    def _compute_lexical_features(self, text):
        tokens = word_tokenize(text.lower())
        tokens_without_sw = [word for word in tokens if word not in self.stop_words]
        
        return {
            "lexical_diversity": len(set(tokens)) / len(tokens) if tokens else 0,
            "stopword_ratio": (len(tokens) - len(tokens_without_sw)) / len(tokens) if tokens else 0,
            "hapax_legomena_ratio": len([word for word, count in Counter(tokens).items() if count == 1]) / len(tokens) if tokens else 0
        }
    
    def _compute_syntactic_features(self, text):
        sentences = sent_tokenize(text)
        
        return {
            "avg_commas_per_sentence": np.mean([sent.count(',') for sent in sentences]),
            "question_ratio": sum(1 for sent in sentences if sent.endswith('?')) / len(sentences) if sentences else 0
        }
    
    def _compute_semantic_features(self, text):
        # TF-IDF analysis
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        tfidf_scores = dict(zip(self.tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens due to model constraints
        
        return {
            "top_tfidf_words": sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            "sentiment_label": sentiment['label'],
            "sentiment_score": sentiment['score']
        }
    
    def get_ml_features(self, analysis_result):
        # Convert analysis results into numerical features suitable for ML models
        return [
            analysis_result["word_count"],
            analysis_result["unique_word_count"],
            analysis_result["sentence_count"],
            analysis_result["avg_word_length"],
            analysis_result["avg_sentence_length"],
            analysis_result["lexical_diversity"],
            analysis_result["stopword_ratio"],
            analysis_result["hapax_legomena_ratio"],
            analysis_result["avg_commas_per_sentence"],
            analysis_result["question_ratio"],
            analysis_result["sentiment_score"]
        ]