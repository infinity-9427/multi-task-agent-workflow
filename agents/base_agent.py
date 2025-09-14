"""
Base Agent class for the multi-agent system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from settings.settings import Settings


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self):
        self.settings = Settings()
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the type/name of the agent"""
        pass