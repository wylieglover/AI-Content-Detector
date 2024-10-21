from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, content: Any) -> Dict[str, Any]:
        pass