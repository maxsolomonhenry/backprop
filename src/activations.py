from backprop.element import Element
import numpy as np

class SigmoidResult(Element):
    _op = 'σ'
    def __init__(self, left):
        super().__init__(self._sigmoid(left._value), left)

    def _grad_fn(self):
        self._left._grad += (self._value * (1.0 - self._value)) * self._grad

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
def sigmoid(x: Element) -> Element:
    return SigmoidResult(x)