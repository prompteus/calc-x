import abc
import sympy

class Gadget(abc.ABC):

    def __init__(self):
        pass

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

    def _float_eval(self, input_str: str) -> float:
        input_str = input_str.replace(",", "")
        expr = sympy.parsing.sympy_parser.parse_expr(input_str, evaluate=True).evalf()
        return float(expr)

    def __call__(self, input_str: str) -> str:
        try:
            input_str = input_str.replace(",", "")
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