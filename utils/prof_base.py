from abc import ABC, abstractmethod

class ProfBase(ABC):
    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError