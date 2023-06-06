from __future__ import annotations

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
        expr = Calculator.evaluate(input_str)
        return float(expr.evalf())

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
    def format_sympy_number(x: sympy.Number, add_approx: bool = True) -> str:
        if isinstance(x, sympy.core.numbers.Integer):
            return Calculator.format_sympy_int(x)
        if isinstance(x, sympy.core.numbers.Float):
            return Calculator.format_sympy_float(x)
        if isinstance(x, sympy.core.numbers.Rational):
            num, den = x.as_numer_denom()
            num = Calculator.format_sympy_number(num, add_approx=False)
            den = Calculator.format_sympy_number(den, add_approx=False)
            string = f"{num}/{den}"
            if add_approx:
                string += f" = around {Calculator.format_sympy_float(x.evalf())}"
            return string
        
        if add_approx:
            return f"{str(x)} = around {Calculator.format_sympy_float(x.evalf())}"
        
        return str(x)
    
    def __call__(self, input_str: str, add_approx: bool = True) -> str:
        try:
            expr = self.evaluate(input_str)
            return self.format_sympy_number(expr, add_approx=add_approx)
        except Exception as e:
            return f"ERROR: {e}"
