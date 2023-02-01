import abc
from chain_gen import get_random_chain, chain_gen

class DataIterator(abc.ABC):
    @abc.abstractmethod
    def __next__(self) -> Tuple[str, str, str]: 
        ...
        
        
