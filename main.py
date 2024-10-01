from analyzers.text_analyzer import TextAnalyzer
from models.federated_ml_model import FederatedMLModel
from engines.decision_engine import DecisionEngine

def main():
    # Initialize the text analyzer, federated model, and decision engine
    text_analyzer = TextAnalyzer()
    federated_model = FederatedMLModel()
    decision_engine = DecisionEngine()
    
    # Sample text for analysis
    sample_text = "Hey! How can I assist you today?"
    
    # Analyze the text
    text_features = text_analyzer.analyze(sample_text)
    
    # Aggregate features
    aggregated_features = federated_model.aggregate_features(text_features)

    # Make a decision
    decision_results = decision_engine.make_decision(aggregated_features)

    print("Aggregated Features:", aggregated_features)
    print("Decision Results:", decision_results)

if __name__ == "__main__":
    main()
