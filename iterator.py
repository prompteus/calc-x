import abc
from typing import Tuple

class DataIterator(abc.ABC):
    @abc.abstractmethod
    def __next__(self) -> Tuple[str, str, str]: 
        ...
        
        
