from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ContentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, content: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_ml_features(self, analysis_result: Dict[str, Any]) -> List[float]:
        pass