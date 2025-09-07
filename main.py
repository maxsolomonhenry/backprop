from __future__ import annotations
import numpy as np

class Element:
    def __init__(self, value):
        self._value = value
        self._grad = 0
        self._left = None
        self._right = None

    def __add__(self, other) -> Element:
        result = Element(self._value + self._get_value(other))
        result._left = self
        result._right = other
        self._grad += 1
        if isinstance(other, Element):
            other._grad += 1
        return result
    
    def __sub__(self, other) -> Element:
        result = Element(self._value - self._get_value(other))
        result._left = self
        result._right = other
        self._grad += 1
        if isinstance(other, Element):
            other._grad += -1
        return result
    
    def __mul__(self, other) -> Element:
        result = Element(self._value * self._get_value(other))
        result._left = self
        result._right = other
        self._grad += self._get_value(other)
        if isinstance(other, Element):
            other._grad += self._value
        return result

    def __truediv__(self, other) -> Element:
        result = Element(self._value / self._get_value(other))
        result._left = self
        result._right = other
        self._grad += 1 / self._get_value(other)
        if isinstance(other, Element):
            other._grad += - self._value /  other._value ** 2
        return result
    
    def __pow__(self, other) -> Element:
        result = Element(self._value ** self._get_value(other))
        result._left = self
        result._right = other
        self._grad += self._get_value(other) * self._value ** (self._get_value(other) - 1)
        if isinstance(other, Element):
            other._grad += result._value * np.log(self._value)
        return result
    
    def __radd__(self, other) -> Element:
        result = Element(self._value + other)
        result._left = other
        result._right = self
        self._grad += 1
        return result
    
    def __rsub__(self, other) -> Element:
        result = Element(other - self._value)
        result._left = other
        result._right = self
        self._grad += -1
        return result
    
    def __rmul__(self, other) -> Element:
        result = Element(self._value * other)
        result._left = other
        result._right = self
        self._grad += other
        return result
    
    def __rtruediv__(self, other) -> Element:
        result = Element(self._get_value(other) / self._value)
        result._left = other
        result._right = self
        self._grad += - other /  self._value ** 2
        return result
    
    def __rpow__(self, other) -> Element:
        result = Element(self._get_value(other) ** self._value)
        result._left = other
        result._right = self
        self._grad += result._value * np.log(other)
        return result
    
    def __pos__(self) -> Element:
        return self
    
    def __neg__(self) -> Element:
        result = Element(-self._value)
        result._left = self
        self._grad += -1
        return result
    
    def __abs__(self) -> Element:
        result = Element(abs(self._value))
        result._left = self
        self._grad += np.sign(self._value)
        return result

    def __repr__(self) -> str:
        return f"Element(value={self._value}, grad={self._grad})"
    
    def _get_value(self, other):
        return other._value if isinstance(other, Element) else other
    
    def backward(self, is_root=True):

        if is_root:
            self._grad = 1

        if isinstance(self._left, Element):
            self._left._grad *= self._grad
            self._left.backward(is_root=False)

        if isinstance(self._right, Element):
            self._right._grad *= self._grad
            self._right.backward(is_root=False)

    def reset(self):
        self._grad = 0

        if isinstance(self._left, Element):
            self._left.reset()

        if isinstance(self._right, Element):
            self._right.reset()


def announce(title, n_repeat=10):
    print("=" * n_repeat, title, "=" * n_repeat)

if __name__ == "__main__":
    x = Element(2)
    y = Element(3)

    print(x._grad, y._grad)

    z = x * y
    a = z ** 2

    print(x._grad, y._grad)

    a.backward()
    print(x._grad, y._grad)
    
    a.reset()
    announce("Reset")

    print(x._grad, y._grad)
    a = ((x * y) ** 2)
    print(x._grad, y._grad)

    a.backward()
    print(x._grad, y._grad)

    announce("c^2 + c, c:=4")
    c = Element(4)
    q = c ** 2 + c

    q._grad = 1
    q.backward()

    print(c._grad)

    x1 = Element(3)
    y1 = Element(4)

    announce("Double element")
    d1 = Element(2)
    d2 = d1 * d1 + d1
    d2.backward()
    print(d1)

    # announce("Bug: Doubled Gradient")
    # z1 = x1 * y1
    # print(x1, y1)

    # z2 = x1 * y1
    # print(x1, y1)
