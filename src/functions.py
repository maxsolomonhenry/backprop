from backprop.element import Element
import numpy as np

class LogResult(Element):
    _op ='ln'
    def __init__(self, left):
        super().__init__(np.log(left._value), left)

    def _grad_fn(self):
        self._left._grad += (1.0 / self._left._value) * self._grad

def log(x: Element) -> Element:
    return LogResult(x)