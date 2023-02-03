import abc
import sympy

class Gadget(abc.ABC):

    bos_template = "<gadget id='%s'>"
    eos_template = "</gadget>"
    response_template = "<output>\n%s\n</output>"

    def __init__(self):
        pass

    def get_gadget_request_bos(self) -> str:
        return self.bos_template % self.gadget_id()

    @staticmethod
    def get_gadget_request_eos() -> str:
        return Gadget.eos_template

    @staticmethod
    def get_gadget_request_eos() -> str:
        return Gadget.eos_template

    @staticmethod
    @abc.abstractmethod
    def gadget_id() -> str:
        ...

    @abc.abstractmethod
    def __call__(self, input_str: str) -> str:
        ...

class Calculator(Gadget):

    @staticmethod
    def gadget_id() -> str:
        return "calculator"

    def __call__(self, input_str: str) -> str:
        try:
            expr = sympy.parsing.sympy_parser.parse_expr(input_str, evaluate=True)
            if isinstance(expr, sympy.core.numbers.Integer):
                string = self.format_sympy_int(expr)
            elif isinstance(expr, sympy.core.numbers.Float):
                string = self.format_sympy_float(expr)
            else:
                string = f"{str(expr)} ~= {self.format_sympy_float(expr.evalf())}"
            return string
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def format_sympy_float(x: sympy.core.numbers.Float) -> str:
        return f"{float(x.evalf()):,.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def format_sympy_int(x: sympy.core.numbers.Integer) -> str:
        return f"{int(x):,}"