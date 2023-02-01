import abc


class Gadget(abc.ABC):

    def __init__(self):
        pass

    def __call__(self, input_str: str) -> str:
        pass


class PythonCLI(Gadget):

    def __call__(self, input_str: str) -> str:
        return exec(input_str)
