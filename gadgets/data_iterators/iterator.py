import abc
import gadgets.datatypes

class DataIterator(abc.ABC):
    @abc.abstractmethod
    def __next__(self) -> gadgets.datatypes.Example:
        ...
