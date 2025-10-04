from backprop.element import Element
import numpy as np

def sigmoid(x : Element) -> Element:
    value = _sigmoid(x._value)
    result = Element(value=value)
    result._left = x

    result._dleft = _sigmoid(x._value) * (1.0 - _sigmoid(x._value))
    return result

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))