# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learning Context
- You are for learning. Don't do code voluntarily. I will ask questions. You can use examples but don't write in my codebase.
- Please do not make suggestions about how I should structure my code or what algorithms I should use. I am learning. Help me with the questions I ask you.

## Codebase Architecture

This is a simple backpropagation implementation for educational purposes. The core architecture consists of:

- `src/element.py`: The main `Element` class that implements automatic differentiation with operator overloading
- `main.py`: Example usage showing a gradient computation bug demonstration
- `tests/test_backprop.py`: Comprehensive test suite with 11 test cases covering various operations

### Key Components

**Element Class (`src/element.py`)**:
- Implements automatic differentiation through operator overloading
- Supports basic arithmetic operations (+, -, *, /, **)
- Uses computational graph with `_left`, `_right` children and local gradients `_dleft`, `_dright`
- Backward pass uses topological traversal of the computation graph

**Import Structure**:
- Import from root: `from element import Element` (main.py style)
- Import from package: `from src.element import Element`

## Common Commands

**Run main example**: `python main.py`
**Run tests**: `python tests/test_backprop.py`  
**Install package**: `pip install -e .` (uses pyproject.toml)

**Dependencies**: Only requires `numpy`