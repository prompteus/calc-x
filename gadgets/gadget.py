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

    @staticmethod
    def _float_eval(input_str: str) -> float:
        expr = sympy.parse_expr(input_str, evaluate=True).evalf()
        return float(expr)

    @staticmethod
    def evaluate(input_str: str) -> sympy.Number:
        return sympy.parse_expr(input_str, evaluate=True)

    @staticmethod
    def format_sympy_float(x: sympy.core.numbers.Float) -> str:
        return f"{float(x):_.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def format_sympy_int(x: sympy.core.numbers.Integer) -> str:
        return f"{int(x):_}"
    
    @staticmethod
    def format_sympy_number(x: sympy.Number) -> str:
        if isinstance(x, sympy.core.numbers.Integer):
            return Calculator.format_sympy_int(x)
        elif isinstance(x, sympy.core.numbers.Float):
            return Calculator.format_sympy_float(x)
        else:
            return f"{str(x)} = around {Calculator.format_sympy_float(x.evalf())}"
    
    def __call__(self, input_str: str) -> str:
        try:
            expr = self.evaluate(input_str)
            return self.format_sympy_number(expr)
        except Exception as e:
            return f"ERROR: {e}"
