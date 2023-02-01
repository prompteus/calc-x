import abc
import subprocess


class Gadget(abc.ABC):

    bos_template = "<gadget %s>"
    eos_template = "</gadget>"

    def __init__(self):
        pass

    def get_gadget_request_bos(self) -> str:
        return self.bos_template % self.gadget_id()

    @staticmethod
    def get_gadget_request_eos() -> str:
        return Gadget.eos_template

    @staticmethod
    @abc.abstractmethod
    def gadget_id() -> str:
        ...

    @abc.abstractmethod
    def setup(self):
        ...

    @abc.abstractmethod
    def __call__(self, input_str: str) -> str:
        ...


class Calculator(Gadget):

    @staticmethod
    def gadget_id() -> str:
        return "calculator"

    def setup(self):
        pass

    def __call__(self, input_str: str) -> str:
        return subprocess.run(["python3", "-c", f"print(eval('{input_str}'))"], capture_output=True).stdout.decode("utf-8")


foo = Calculator()
