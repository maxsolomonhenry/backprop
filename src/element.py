from __future__ import annotations
import numpy as np

class Element:
    def __init__(self, value):
        self._value = value
        self._grad = 0
        self._left = None
        self._right = None
        self._op = None

    def __add__(self, other) -> Element:
        result = self._binary_result(other, self._value + self._get_value(other))

        result._dleft = 1
        result._dright = 1
        return result
    
    def __sub__(self, other) -> Element:
        result = self._binary_result(other, self._value - self._get_value(other))

        result._dleft = 1
        result._dright = -1
        return result
    
    def __mul__(self, other) -> Element:
        result = self._binary_result(other, self._value * self._get_value(other))

        result._dleft = self._get_value(other)
        result._dright = self._value
        return result

    def __truediv__(self, other) -> Element:
        result = self._binary_result(other, self._value / self._get_value(other))

        result._dleft = 1 / self._get_value(other)
        result._dright = - self._value /  self._get_value(other) ** 2
        return result
    
    def __pow__(self, other) -> Element:
        result = self._binary_result(other, self._value ** self._get_value(other))

        result._dleft = self._get_value(other) * self._value ** (self._get_value(other) - 1)
        result._dright = 0 if self._value == 0 else result._value * np.log(self._value)
        return result
    
    def __radd__(self, other) -> Element:
        result = self._binary_result(other, self._value + other, is_reverse=True)

        result._dright = 1
        return result
    
    def __rsub__(self, other) -> Element:
        result = self._binary_result(other, other - self._value, is_reverse=True)

        result._dright = -1
        return result
    
    def __rmul__(self, other) -> Element:
        result = self._binary_result(other, self._value * other, is_reverse=True)

        result._dright = other
        return result
    
    def __rtruediv__(self, other) -> Element:
        result = self._binary_result(other, other / self._value, is_reverse=True)

        result._dright = - other /  self._value ** 2 
        return result
    
    def __rpow__(self, other) -> Element:
        result = self._binary_result(other, other ** self._value, is_reverse=True)

        result._dright = 0 if result._value == 0 else result._value * np.log(other)
        return result
    
    def __pos__(self) -> Element:
        return self
    
    def __neg__(self) -> Element:
        result = Element(-self._value)
        result._left = self

        result._dleft = -1
        return result
    
    def __abs__(self) -> Element:
        result = Element(abs(self._value))
        result._left = self
        result._dleft = np.sign(self._value)
        return result

    def __repr__(self) -> str:
        return f"Element(value={self._value}, grad={self._grad})"
    
    def _get_value(self, other):
        return other._value if isinstance(other, Element) else other
    
    def _binary_result(self, other, value, is_reverse=False):
        result = Element(value)

        if is_reverse:
            result._left = other
            result._right = self
        else:
            result._left = self
            result._right = other
        return result
    
    def _traverse(self, node_order=None):
        if node_order is None:
            node_order = []

        if isinstance(self._left, Element):
            self._left._traverse(node_order)
        if isinstance(self._right, Element):
            self._right._traverse(node_order)

        if self not in node_order:
            node_order.append(self)
        return node_order

    def backward(self):

        self.reset()
        self._grad = 1

        node_order = self._traverse()
        for node in reversed(node_order):
            if isinstance(node._left, Element):
                node._left._grad += node._dleft * node._grad
            if isinstance(node._right, Element):
                node._right._grad += node._dright * node._grad

    def reset(self):
        self._grad = 0

        if isinstance(self._left, Element):
            self._left.reset()

        if isinstance(self._right, Element):
            self._right.reset()