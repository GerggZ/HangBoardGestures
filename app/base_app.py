from abc import ABC, abstractmethod

class BaseApp(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
