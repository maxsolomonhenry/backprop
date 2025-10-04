# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal automatic differentiation (backpropagation) engine implemented from scratch in Python. The core abstraction is the `Element` class which builds computational graphs dynamically during forward operations and computes gradients via reverse-mode automatic differentiation.

## Architecture

### Core Components

**Element Class** (`src/element.py`)
- The fundamental building block that wraps scalar values and tracks gradient information
- Implements operator overloading for arithmetic operations (`+`, `-`, `*`, `/`, `**`)
- Maintains a computational graph implicitly through `_left` and `_right` child references
- Stores local derivatives (`_dleft`, `_dright`) during forward pass for efficiency
- The `backward()` method performs topological traversal and gradient accumulation via the chain rule

**Computational Graph Structure**
- Built implicitly during forward operations
- Binary operations create nodes with `_left` and `_right` children
- Unary operations (like activations) create nodes with only `_left` child
- The graph is traversed in reverse topological order during backpropagation

**Key Design Pattern**
- Each operation stores its local partial derivatives during the forward pass (e.g., `_dleft`, `_dright`)
- The `backward()` method performs a topological sort via `_traverse()`, then accumulates gradients in reverse order
- Gradients accumulate additively, which correctly handles shared nodes (when a variable is used multiple times)

**Activation Functions** (`src/activations.py`)
- Follow the pattern: create result `Element`, set `_left` to input, compute `_dleft` (local derivative)
- Currently implements `sigmoid(x)` with derivative `σ(x)(1-σ(x))`

### Project Structure

```
backprop/
├── src/
│   ├── element.py        # Core Element class with autodiff
│   ├── activations.py    # Activation functions (sigmoid, etc.)
│   └── __init__.py
├── tests/
│   └── test_backprop.py  # Comprehensive test suite (27 tests)
├── main.py               # Example demonstrating shared node behavior
└── notebook.ipynb        # Interactive development notebook
```

## Development Commands

### Installation
```bash
pip install -e .
```

### Running Tests
```bash
python tests/test_backprop.py
```

The test suite includes 27 tests covering:
- Basic operations (addition, multiplication, division, powers)
- Chain rule and composition
- Edge cases (zero, one, negative numbers, small numbers)
- Shared nodes and diamond graphs (where gradients must accumulate correctly)
- Deep nesting and wide expressions

### Interactive Development
The `notebook.ipynb` uses autoreload for iterative development:
```python
%load_ext autoreload
%autoreload 2
from src.element import Element
from src.activations import sigmoid
```

## Implementation Notes

### Adding New Activation Functions
Follow this pattern (see `src/activations.py`):
```python
def new_activation(x: Element) -> Element:
    value = _compute_activation(x._value)
    result = Element(value=value)
    result._left = x
    result._dleft = _compute_derivative(x._value)  # Local derivative
    return result
```

### Shared Nodes / Gradient Accumulation
The engine correctly handles cases where the same `Element` appears multiple times in a computation graph. Gradients are accumulated additively using `+=` in `backward()`, which implements the multivariate chain rule correctly.

### Known Issues
The `main.py` file demonstrates a previously fixed bug where shared intermediate nodes weren't handling gradients correctly. This has been resolved by the topological traversal in `_traverse()`.
