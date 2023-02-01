import abc
import subprocess

class Gadget(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def setup(self):
        ...

    @abc.abstractmethod
    def __call__(self, input_str: str) -> str:
        ...


class Calculator(Gadget):

    def setup(self):
        ...

    def __call__(self, input_str: str) -> str:
        return subprocess.run(["python3", "-c", f"print(eval('{input_str}'))"], capture_output=True).stdout.decode("utf-8")


foo = Calculator()
