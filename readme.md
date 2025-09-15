# Backprop
I am making a backprop engine. Whoopdee doo. 

## Element
The basic class is called `Element` which implicitly builds a computational 
graph during operations (add, subtract, etc.). The backward pass is initiated 
with `.backword()` and flushes the gradient back through a topological sort. 
We're learning a lot, aren't we. 

## Install
This package is pip installable. Hurray.
```
pip install -e .
```

## Tests
The file `test_backprop.py` runs basic unit tests on the element class.
