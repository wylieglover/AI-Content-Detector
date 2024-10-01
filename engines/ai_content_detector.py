from analyzers.text_analyzer import TextAnalyzer
from analyzers.video_analyzer import VideoAnalyzer
from analyzers.audio_analyzer import AudioAnalyzer
from analyzers.image_analyzer import ImageAnalyzer
from typing import Dict, Any

class AIContentDetector:
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.image_analyzer = ImageAnalyzer()

    def analyze_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the input content based on its type.
        :param content: A dictionary that contains different types of content like 'text', 'video', 'audio', 'image'
        :return: A dictionary with analysis results from all analyzers.
        """
        analysis_results = {}

        if 'text' in content:
            print("Analyzing text...")
            analysis_results['text'] = self.text_analyzer.analyze(content['text'])

        if 'video' in content:
            print("Analyzing video...")
            analysis_results['video'] = self.video_analyzer.analyze(content['video'])

        if 'audio' in content:
            print("Analyzing audio...")
            analysis_results['audio'] = self.audio_analyzer.analyze(content['audio'])

        if 'image' in content:
            print("Analyzing image...")
            analysis_results['image'] = self.image_analyzer.analyze(content['image'])

        return analysis_results

    def get_features_for_federated_model(self, analysis_results: Dict[str, Any]) -> Dict[str, list]:
        """
        Extract machine learning features from the analysis results.
        :param analysis_results: A dictionary of analysis results.
        :return: A dictionary of ML features for each content type.
        """
        ml_features = {}

        if 'text' in analysis_results:
            ml_features['text'] = self.text_analyzer.get_ml_features(analysis_results['text'])

        if 'video' in analysis_results:
            ml_features['video'] = self.video_analyzer.get_ml_features(analysis_results['video'])

        if 'audio' in analysis_results:
            ml_features['audio'] = self.audio_analyzer.get_ml_features(analysis_results['audio'])

        if 'image' in analysis_results:
            ml_features['image'] = self.image_analyzer.get_ml_features(analysis_results['image'])

        return ml_features
