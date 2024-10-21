
class DecisionEngine:
    def make_decision(self, aggregated_results):
        # Use AI-generated probabilities from MLAnalyzer
        probabilities = [
            value for key, value in aggregated_results.items()
            if 'ai_generated_probability' in key
        ]
        avg_probability = sum(probabilities) / len(probabilities)
        blockchain_verified = aggregated_results.get('blockchain_verified')
        
        # Consider blockchain verification
        if blockchain_verified:
            decision = 'Authentic (Blockchain Verified)'
        else:
            decision = 'AI-Generated' if avg_probability > 0.5 else 'Authentic'

        return decision
