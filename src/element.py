from __future__ import annotations
import numpy as np


def _val(obj):
    return obj._value if isinstance(obj, Element) else obj

class Element:
    _op = 'None'
    def __init__(self, value, left=None, right=None):
        self._value = value
        self._left = self._ensure_element(left)
        self._right = self._ensure_element(right)

        self._grad = 0

    @staticmethod
    def _ensure_element(obj):
        if obj is None:
            return obj
        return obj if isinstance(obj, Element) else Element(obj)

    def _grad_fn(self):
        pass

    def __add__(self, other) -> Element:
        return AddResult(self, other)
        
    def __sub__(self, other) -> Element:
        return SubtractResult(self, other)
    
    def __mul__(self, other) -> Element:
        return MultiplyResult(self, other)

    def __truediv__(self, other) -> Element:
        return DivideResult(self, other)
    
    def __pow__(self, other) -> Element:
        return PowerResult(self, other)
    
    def __radd__(self, other) -> Element:
        return AddResult(other, self)
    
    def __rsub__(self, other) -> Element:
        return SubtractResult(other, self)
    
    def __rmul__(self, other) -> Element:
        return MultiplyResult(other, self)
    
    def __rtruediv__(self, other) -> Element:
        return DivideResult(other, self)
    
    def __rpow__(self, other) -> Element:
        return PowerResult(other, self)
    
    def __pos__(self) -> Element:
        return self
    
    def __neg__(self) -> Element:
        return MultiplyResult(self, -1)
    
    def __abs__(self) -> Element:
        return AbsoluteResult(self)

    def __repr__(self) -> str:
        return f"Element(value={self._value}, grad={self._grad}, op={self._op})"
    
    def _traverse(self, node_order=None):
        if node_order is None:
            node_order = []

        if isinstance(self._left, Element):
            self._left._traverse(node_order)
        if isinstance(self._right, Element):
            self._right._traverse(node_order)

        if self not in node_order:
            node_order.append(self)

    def backward(self):
        self._grad = 1

        node_order = []
        self._traverse(node_order)
        
        for node in reversed(node_order):
            node._grad_fn()

    def reset(self):
        self._grad = 0

        if isinstance(self._left, Element):
            self._left.reset()

        if isinstance(self._right, Element):
            self._right.reset()


class AddResult(Element):
    _op = '+'
    def __init__(self, left, right):
        super().__init__(_val(left) + _val(right), left, right)

    def _grad_fn(self):
        self._left._grad += self._grad
        self._right._grad += self._grad


class SubtractResult(Element):
    _op = '-'
    def __init__(self, left, right):
        super().__init__(_val(left) - _val(right), left, right)

    def _grad_fn(self):
        self._left._grad += self._grad
        self._right._grad -= self._grad


class MultiplyResult(Element):
    _op = 'x'
    def __init__(self, left, right):
        super().__init__(_val(left) * _val(right), left, right)

    def _grad_fn(self):
        self._left._grad += self._right._value * self._grad
        self._right._grad += self._left._value * self._grad


class DivideResult(Element):
    _op = '/'
    def __init__(self, left, right):
        super().__init__(_val(left) / _val(right), left, right)

    def _grad_fn(self):
        self._left._grad += 1.0 / self._right._value * self._grad
        self._right._grad -= self._left._value / self._right._value ** 2 * self._grad


class PowerResult(Element):
    _op = '^'
    def __init__(self, left, right):
        super().__init__(_val(left) ** _val(right), left, right)

    def _grad_fn(self):
        self._left._grad += self._right._value * self._left._value ** (self._right._value - 1) * self._grad
        self._right._grad += (0 if self._left._value == 0 else self._value * np.log(self._left._value)) * self._grad

class AbsoluteResult(Element):
    _op = '| |'
    def __init__(self, left):
        super().__init__(np.abs(_val(left)), left)

    def _grad_fn(self):
        self._left._grad += np.sign(self._left._value) * self._grad