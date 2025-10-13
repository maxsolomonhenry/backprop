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

## Bugs
Known bugs (last updated 10/13/25)
- The transpose creates a new node, which wipes the gradient. This works in 
the computational graph but if you just want to have a look at an Element as a 
transposed item it leads to counterintuitive results.

## Learnings
A few personal a-ha moments along the way:
- You don't store the Jacobian (i.e., gradient wrt adjacent node), rather each
`Element` stores the gradient of the "loss," assumed to be a scalar, wrt to itself.
As a result, the local gradient will have, by definition, the same shape as the 
value it is storing (dL/dW will have the shape of W if L is a scalar). Hence the 
importance of defining a gradient _function_ which can update the parents. This
prevents the need to store hugely high-dimensional intermediate tensors (e.g., 
`dAX/dA` is a four-dimensional tensor, assuming A and X are vectors -- instead, just
calculate `dL/dA := dL/dAX * dAX/dA`).
- Einstein notation will save your life when determining the backwards pass.
- A gotcha: if the same term occurs twice in the same figure (e.g., `x_i x_i` as in `x.T x`), don't forget to add the gradient twice as dictated by the multi-variate chain rule. This is not immediately obvious.
- Topological sort first, this is necessary for the gradients to accumulate properly.
- An element-wise operation in the forward pass becomes an element-wise operation in the backward pass.

## TODO
-[ ] `sum()` and `mean()` functions to get to a scalar more easily.
-[ ] simple scalar mults, adds, divisions, etc that aren't tracked in backprop
-[ ] finish vectorizing operation nodes in `Element.py`
-[ ] expand to batches
-[ ] train POC (MNIST)