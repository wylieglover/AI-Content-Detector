from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline
from .base_analyzer import BaseAnalyzer
import spacy
import logging

class TextAnalyzer(BaseAnalyzer):
    def __init__(self, fit_tfidf=True, corpus=None, fit_ngrams=True):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
        self.model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            device=0,
            truncation=True,       # Enable truncation
            max_length=512         # Set maximum token length
        )
        self.nlp = spacy.load('en_core_web_sm')
        
        self.bigram_encoder = OneHotEncoder(handle_unknown='ignore')
        self.trigram_encoder = OneHotEncoder(handle_unknown='ignore')
        
        if fit_ngrams and corpus:
            bigrams = [bigram for text in corpus for bigram in self._extract_bigrams(text)]
            trigrams = [trigram for text in corpus for trigram in self._extract_trigrams(text)]
            self.bigram_encoder.fit(np.array(bigrams).reshape(-1,1))
            self.trigram_encoder.fit(np.array(trigrams).reshape(-1,1))
            logging.info("N-gram encoders fitted on the corpus.")
            
        if fit_tfidf and corpus:
            self.tfidf_vectorizer.fit(corpus)
            logging.info("TF-IDF vectorizer fitted on the corpus.")
        
    def analyze_batch(self, texts, batch_size=32):
        """
        Analyze a batch of texts for better performance using nlp.pipe.
        """
        features_list = []
        sentiments = self.sentiment_analyzer(texts, batch_size=batch_size)
        
        for doc, sentiment in zip(self.nlp.pipe(texts, batch_size=batch_size, disable=["ner"]), sentiments):
            tokens = [token.text for token in doc if not token.is_space]
            sentences = list(doc.sents)
            
            features = {}
            features.update(self._compute_basic_stats(doc.text))
            features.update(self._compute_lexical_features(doc.text))
            features.update(self._compute_syntactic_features(doc.text))
            features.update(self._compute_semantic_features(doc.text))
            features.update(self._compute_punctuation_usage(doc.text))
            features.update(self._compute_vocabulary_richness(doc.text))
            features.update(self._compute_pos_tags(doc.text))
            features.update(self._compute_syntax_patterns(doc.text))
            features.update(self._compute_ngrams(doc.text))
            
            features_list.append(features)
        
        return features_list
    
    def analyze(self, text):
        """
        Analyze a single text by utilizing the batch method.
        """
        return self.analyze_batch([text], batch_size=1)[0]
    
    def _compute_basic_stats(self, text):
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
        avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
        
        # Sentence length variability
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Word length variability
        word_lengths = [len(word) for word in tokens]
        word_length_std = np.std(word_lengths) if word_lengths else 0
        
        return {
            "word_count": len(tokens),
            "unique_word_count": len(set(tokens)),
            "sentence_count": len(sentences),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_std": sentence_length_std,
            "word_length_std": word_length_std
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
        
        avg_commas_per_sentence = np.mean([sent.count(',') for sent in sentences]) if sentences else 0
        question_ratio = sum(1 for sent in sentences if sent.endswith('?')) / len(sentences) if sentences else 0
        
        return {
            "avg_commas_per_sentence": avg_commas_per_sentence,
            "question_ratio": question_ratio
        }
    
    def _compute_semantic_features(self, text):
        # TF-IDF analysis using pre-fitted vectorizer
        tfidf_matrix = self.tfidf_vectorizer.transform([text])
        tfidf_scores = dict(zip(self.tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
        
        # Sentiment analysis with proper truncation
        sentiment = self.sentiment_analyzer(text)[0]  # The pipeline handles truncation
        
        return {
            "top_tfidf_words": sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            "sentiment_label": sentiment['label'],
            "sentiment_score": sentiment['score']
        }
    
    def _compute_punctuation_usage(self, text):
        punctuation_marks = ['.', ',', ';', ':', '!', '?', '-', '"', "'", '(', ')']
        total_punctuations = sum(text.count(p) for p in punctuation_marks)
        punctuation_counts = {f"punct_{p}": text.count(p) for p in punctuation_marks}
        
        # Frequency of each punctuation mark per 100 words
        tokens = word_tokenize(text)
        word_count = len(tokens) if tokens else 1  # Avoid division by zero
        punctuation_freq = {k: (v / word_count) * 100 for k, v in punctuation_counts.items()}
        
        return {
            "total_punctuations": total_punctuations,
            "punctuation_variety": sum(1 for v in punctuation_counts.values() if v > 0),
            **punctuation_freq
        }
    
    def _compute_vocabulary_richness(self, text):
        tokens = word_tokenize(text.lower())
        N = len(tokens)
        unique_tokens = set(tokens)
        V = len(unique_tokens)
        ttr = V / N if N > 0 else 0  # Type-Token Ratio

        # Advanced measures (e.g., Honore's Statistic)
        if N == 0 or V == 0:
            honore_statistic = 0.0
        else:
            hapax_legomena = [word for word, count in Counter(tokens).items() if count == 1]
            V1 = len(hapax_legomena)
            ratio = V1 / V
            denominator = 1 - ratio
            if denominator <= 0:
                honore_statistic = 0.0
            else:
                honore_statistic = (100 * np.log(N)) / denominator
        return {
            "type_token_ratio": ttr,
            "honore_statistic": honore_statistic
        }
    
    def _compute_pos_tags(self, text):
        # Using SpaCy for POS tagging
        doc = self.nlp(text)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        pos_distribution = {f"pos_{pos}": count / total_tokens for pos, count in pos_counts.items()}
        
        return pos_distribution
    
    def _compute_syntax_patterns(self, text):
        # Using SpaCy for dependency parsing
        doc = self.nlp(text)
        dependency_counts = Counter([token.dep_ for token in doc])
        total_tokens = len(doc)
        
        dependency_distribution = {f"dep_{dep}": count / total_tokens for dep, count in dependency_counts.items()}
        
        return dependency_distribution
    
    def _extract_bigrams(self, text):
        tokens = word_tokenize(text.lower())
        return list(ngrams(tokens, 2))
    
    def _extract_trigrams(self, text):
        tokens = word_tokenize(text.lower())
        return list(ngrams(tokens, 3))
    
    def _compute_ngrams(self, text):
        tokens = word_tokenize(text.lower())
        bigrams = self._extract_bigrams(text)
        trigrams = self._extract_trigrams(text)
        
        if bigrams:
            bigram_encoded = self.bigram_encoder.transform(np.array(bigrams).reshape(-1,1)).sum(axis=0)
        else:
            bigram_encoded = np.zeros(len(self.bigram_encoder.get_feature_names_out()), dtype=np.float32)
        
        if trigrams:
            trigram_encoded = self.trigram_encoder.transform(np.array(trigrams).reshape(-1,1)).sum(axis=0)
        else:
            trigram_encoded = np.zeros(len(self.trigram_encoder.get_feature_names_out()), dtype=np.float32)
        
        return {
            **{f"bigram_{i}": val for i, val in enumerate(bigram_encoded)},
            **{f"trigram_{i}": val for i, val in enumerate(trigram_encoded)}
        }