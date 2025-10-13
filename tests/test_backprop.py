#!/usr/bin/env python3

from backprop.element import Element
from backprop.activations import sigmoid
import numpy as np

def test_case(name, computation, variables, expected_grads, tolerance=1e-6):
    """Test a single case and report results"""
    print(f"\n=== {name} ===")

    # Reset all variables
    for var in variables:
        var._grad = np.zeros_like(var._value)

    # Run computation and backward pass
    result = computation()
    result.backward()

    # Check results
    all_passed = True
    for i, (var, expected) in enumerate(zip(variables, expected_grads)):
        actual = var._grad
        # Handle both scalar and array grads
        if isinstance(actual, np.ndarray) and actual.size == 1:
            actual_val = actual.item()
            expected_val = expected if not isinstance(expected, np.ndarray) else expected.item()
            passed = abs(actual_val - expected_val) < tolerance
            status = "✓" if passed else "✗"
            print(f"  Var {i+1}: expected={expected_val:.6f}, actual={actual_val:.6f} {status}")
        else:
            # For arrays, use np.allclose
            passed = np.allclose(actual, expected, atol=tolerance)
            status = "✓" if passed else "✗"
            print(f"  Var {i+1}: {status}")
            if not passed:
                print(f"    Expected: {expected}")
                print(f"    Actual: {actual}")
        if not passed:
            all_passed = False

    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def run_tests():
    """Run all test cases"""
    print("Testing Backprop Engine")
    print("=" * 50)
    
    passed = 0
    total = 0
    
    # Test 1: Simple multiplication - d/dx(xy) = y, d/dy(xy) = x
    x1, y1 = Element(3), Element(4)
    total += 1
    if test_case("Simple multiplication: f(x,y) = xy", 
                 lambda: x1 * y1, [x1, y1], [4, 3]):
        passed += 1
    
    # Test 2: Quadratic - d/dx(x²) = 2x
    x2 = Element(5)
    total += 1
    if test_case("Quadratic: f(x) = x²", 
                 lambda: x2 ** 2, [x2], [10]):
        passed += 1
    
    # Test 3: Chain rule - d/dx((xy)²) = 2xy·y = 2xy², d/dy((xy)²) = 2xy·x = 2x²y
    x3, y3 = Element(2), Element(3)
    total += 1
    if test_case("Chain rule: f(x,y) = (xy)²", 
                 lambda: (x3 * y3) ** 2, [x3, y3], [36, 24]):
        passed += 1
    
    # Test 4: Addition - d/dx(x² + x) = 2x + 1
    x4 = Element(4)
    total += 1
    if test_case("Addition: f(x) = x² + x", 
                 lambda: x4 ** 2 + x4, [x4], [9]):
        passed += 1
    
    # Test 5: Subtraction - d/dx(x² - 2x) = 2x - 2
    x5 = Element(3)
    total += 1
    if test_case("Subtraction: f(x) = x² - 2x", 
                 lambda: x5 ** 2 - 2 * x5, [x5], [4]):
        passed += 1
    
    # Test 6: Division - d/dx(x/y) = 1/y, d/dy(x/y) = -x/y²
    x6, y6 = Element(6), Element(2)
    total += 1
    if test_case("Division: f(x,y) = x/y", 
                 lambda: x6 / y6, [x6, y6], [0.5, -1.5]):
        passed += 1
    
    # Test 7: Complex expression - d/dx((x+1)²) = 2(x+1)
    x7 = Element(2)
    total += 1
    if test_case("Complex: f(x) = (x+1)²", 
                 lambda: (x7 + 1) ** 2, [x7], [6]):
        passed += 1
    
    # Test 8: Negative values
    x8 = Element(-2)
    total += 1
    if test_case("Negative: f(x) = x²", 
                 lambda: x8 ** 2, [x8], [-4]):
        passed += 1
    
    # Test 9: Fractional power - d/dx(x^0.5) = 0.5 * x^(-0.5)
    x9 = Element(4)
    total += 1
    if test_case("Fractional power: f(x) = x^0.5", 
                 lambda: x9 ** 0.5, [x9], [0.25]):
        passed += 1
    
    # Test 10: Multiple operations - d/dx(x²y + xy²) = 2xy + y², d/dy(x²y + xy²) = x² + 2xy
    x10, y10 = Element(2), Element(3)
    total += 1
    if test_case("Multi-op: f(x,y) = x²y + xy²", 
                 lambda: x10**2 * y10 + x10 * y10**2, [x10, y10], [21, 16]):
        passed += 1
    
    # Test 11: Deep chain rule with shared nodes
    x11 = Element(2)
    y11 = x11 * x11
    z11 = y11 * y11
    total += 1
    # dz/dx = 4x^3 = 4 * 2^3 = 32
    if test_case("Deep shared node: f(x) = (x²)²",
                 lambda: z11, [x11], [32]):
        passed += 1
    
    # EDGE CASE TESTS
    print(f"\n{'=' * 20} EDGE CASES {'=' * 20}")
    
    # Test 12: Operations with zero
    x12 = Element(5)
    total += 1
    if test_case("Zero multiplication: f(x) = x * 0", 
                 lambda: x12 * 0, [x12], [0]):
        passed += 1
    
    # Test 13: Zero division (dividing by variable that equals zero would be problematic, so test 0/x)
    x13 = Element(3)
    total += 1
    if test_case("Zero division: f(x) = 0 / x", 
                 lambda: 0 / x13, [x13], [0]):
        passed += 1
    
    # Test 14: Power with zero exponent - d/dx(x^0) = 0
    x14 = Element(7)
    total += 1
    if test_case("Zero exponent: f(x) = x^0", 
                 lambda: x14 ** 0, [x14], [0]):
        passed += 1
    
    # Test 15: Power with zero base - d/dx(0^x) = 0 (for x > 0)
    x15 = Element(2)
    total += 1
    if test_case("Zero base: f(x) = 0^x", 
                 lambda: 0 ** x15, [x15], [0]):
        passed += 1
    
    # Test 16: Operations with one
    x16 = Element(4)
    total += 1
    if test_case("One multiplication: f(x) = x * 1", 
                 lambda: x16 * 1, [x16], [1]):
        passed += 1
    
    # Test 17: Division by one
    x17 = Element(6)
    total += 1
    if test_case("Division by one: f(x) = x / 1", 
                 lambda: x17 / 1, [x17], [1]):
        passed += 1
    
    # Test 18: One to variable power - d/dx(1^x) = 0
    x18 = Element(3)
    total += 1
    if test_case("One to power: f(x) = 1^x", 
                 lambda: 1 ** x18, [x18], [0]):
        passed += 1
    
    # Test 19: Diamond graph - variable feeds into multiple paths that reconverge
    x19 = Element(2)
    a19 = x19 + 1  # a = x + 1
    b19 = x19 * 2  # b = 2x  
    c19 = a19 * b19  # c = (x + 1)(2x) = 2x² + 2x
    total += 1
    # dc/dx = 4x + 2 = 4(2) + 2 = 10
    if test_case("Diamond graph: f(x) = (x+1)(2x)", 
                 lambda: c19, [x19], [10]):
        passed += 1
    
    # Test 20: Very deep nesting
    x20 = Element(2)
    result20 = ((x20 + 1) * 2 - 1) ** 2
    total += 1
    # f(x) = ((x+1)*2-1)² = (2x+2-1)² = (2x+1)²
    # df/dx = 2(2x+1)*2 = 4(2x+1) = 4(5) = 20
    if test_case("Deep nesting: f(x) = ((x+1)*2-1)²", 
                 lambda: result20, [x20], [20]):
        passed += 1
    
    # Test 21: Wide expression with many variables
    vars21 = [Element(i) for i in range(1, 6)]  # x1=1, x2=2, x3=3, x4=4, x5=5
    result21 = vars21[0] + vars21[1] + vars21[2] + vars21[3] + vars21[4]
    total += 1
    # f(x1,x2,x3,x4,x5) = x1+x2+x3+x4+x5, all gradients = 1
    if test_case("Wide sum: f(x1,x2,x3,x4,x5) = x1+x2+x3+x4+x5", 
                 lambda: result21, vars21, [1, 1, 1, 1, 1]):
        passed += 1
    
    # Test 22: Complex sharing pattern
    x22, y22 = Element(2), Element(3)
    a22 = x22 * y22     # a = xy = 6
    b22 = y22 * x22     # b = yx = 6 (same as a, but different path)
    c22 = a22 + b22     # c = xy + yx = 2xy
    total += 1
    # dc/dx = 2y = 6, dc/dy = 2x = 4
    if test_case("Complex sharing: f(x,y) = xy + yx", 
                 lambda: c22, [x22, y22], [6, 4]):
        passed += 1
    
    # Test 23: Triple sharing - same variable used in three paths
    x23 = Element(2)
    a23 = x23 ** 2      # a = x²
    b23 = x23 ** 3      # b = x³  
    c23 = x23 ** 4      # c = x⁴
    result23 = a23 + b23 + c23  # f = x² + x³ + x⁴
    total += 1
    # df/dx = 2x + 3x² + 4x³ = 2(2) + 3(4) + 4(8) = 4 + 12 + 32 = 48
    if test_case("Triple sharing: f(x) = x² + x³ + x⁴", 
                 lambda: result23, [x23], [48]):
        passed += 1
    
    # Test 24: Negative numbers edge cases
    x24 = Element(-3)
    total += 1
    if test_case("Negative cubic: f(x) = x³", 
                 lambda: x24 ** 3, [x24], [27]):  # d/dx(x³) = 3x² = 3*9 = 27
        passed += 1
    
    # Test 25: Very small numbers (numerical stability)
    x25 = Element(1e-10)
    total += 1
    if test_case("Small numbers: f(x) = x²", 
                 lambda: x25 ** 2, [x25], [2e-10]):
        passed += 1
    
    # Test 26: Nested sharing - shared intermediate results
    x26 = Element(2)
    shared26 = x26 * x26    # shared = x²
    left26 = shared26 + 1   # left = x² + 1
    right26 = shared26 * 2  # right = 2x²
    result26 = left26 * right26  # result = (x² + 1)(2x²)
    total += 1
    # f = (x² + 1)(2x²) = 2x⁴ + 2x²
    # df/dx = 8x³ + 4x = 8(8) + 4(2) = 64 + 8 = 72
    if test_case("Nested sharing: f(x) = (x² + 1)(2x²)", 
                 lambda: result26, [x26], [72]):
        passed += 1
    
    # Test 27: Chain of operations all using same variable
    x27 = Element(2)
    total += 1
    # f = x + x² + x³ + x⁴ + x⁵
    # df/dx = 1 + 2x + 3x² + 4x³ + 5x⁴ = 1 + 4 + 12 + 32 + 80 = 129
    if test_case("Long chain: f(x) = x + x² + x³ + x⁴ + x⁵",
                 lambda: x27 + x27**2 + x27**3 + x27**4 + x27**5, [x27], [129]):
        passed += 1

    # SIGMOID ACTIVATION TESTS
    print(f"\n{'=' * 20} SIGMOID TESTS {'=' * 20}")

    # Test 28: Basic sigmoid - σ'(x) = σ(x)(1-σ(x))
    x28 = Element(0)
    total += 1
    # σ(0) = 0.5, σ'(0) = 0.5 * 0.5 = 0.25
    if test_case("Sigmoid at zero: f(x) = σ(x)",
                 lambda: sigmoid(x28), [x28], [0.25]):
        passed += 1

    # Test 29: Sigmoid with positive value
    x29 = Element(2)
    sig_val = 1 / (1 + np.exp(-2))  # σ(2) ≈ 0.8808
    expected_grad = sig_val * (1 - sig_val)  # ≈ 0.1050
    total += 1
    if test_case("Sigmoid positive: f(x) = σ(x)",
                 lambda: sigmoid(x29), [x29], [expected_grad]):
        passed += 1

    # Test 30: Sigmoid with negative value
    x30 = Element(-2)
    sig_val = 1 / (1 + np.exp(2))  # σ(-2) ≈ 0.1192
    expected_grad = sig_val * (1 - sig_val)  # ≈ 0.1050
    total += 1
    if test_case("Sigmoid negative: f(x) = σ(x)",
                 lambda: sigmoid(x30), [x30], [expected_grad]):
        passed += 1

    # Test 31: Sigmoid composition - d/dx(σ(x²)) = σ'(x²) * 2x
    x31 = Element(1)
    total += 1
    # x² = 1, σ(1) ≈ 0.7311, σ'(1) ≈ 0.1966
    # df/dx = σ'(1) * 2x = 0.1966 * 2 ≈ 0.3932
    sig_val = 1 / (1 + np.exp(-1))
    expected_grad = sig_val * (1 - sig_val) * 2 * 1
    if test_case("Sigmoid composition: f(x) = σ(x²)",
                 lambda: sigmoid(x31 ** 2), [x31], [expected_grad]):
        passed += 1

    # Test 32: Multiple sigmoids - d/dx(σ(x) + σ(2x))
    x32 = Element(1)
    total += 1
    # For σ(x): σ(1) ≈ 0.7311, σ'(1) ≈ 0.1966
    # For σ(2x): σ(2) ≈ 0.8808, σ'(2) ≈ 0.1050, chain rule: * 2 = 0.2100
    sig1 = 1 / (1 + np.exp(-1))
    grad1 = sig1 * (1 - sig1)
    sig2 = 1 / (1 + np.exp(-2))
    grad2 = sig2 * (1 - sig2) * 2
    expected_grad = grad1 + grad2
    if test_case("Double sigmoid: f(x) = σ(x) + σ(2x)",
                 lambda: sigmoid(x32) + sigmoid(2 * x32), [x32], [expected_grad]):
        passed += 1

    # Test 33: Sigmoid with multiplication - d/dx(x * σ(x))
    x33 = Element(2)
    total += 1
    # f(x) = x * σ(x)
    # df/dx = σ(x) + x * σ'(x)
    sig_val = 1 / (1 + np.exp(-2))
    sig_deriv = sig_val * (1 - sig_val)
    expected_grad = sig_val + 2 * sig_deriv
    if test_case("Sigmoid product: f(x) = x * σ(x)",
                 lambda: x33 * sigmoid(x33), [x33], [expected_grad]):
        passed += 1

    # ABSOLUTE VALUE TESTS
    print(f"\n{'=' * 20} ABSOLUTE VALUE TESTS {'=' * 20}")

    # Test 34: Absolute value of positive - d/dx(|x|) = 1 for x > 0
    x34 = Element(3)
    total += 1
    if test_case("Abs positive: f(x) = |x|",
                 lambda: abs(x34), [x34], [1]):
        passed += 1

    # Test 35: Absolute value of negative - d/dx(|x|) = -1 for x < 0
    x35 = Element(-3)
    total += 1
    if test_case("Abs negative: f(x) = |x|",
                 lambda: abs(x35), [x35], [-1]):
        passed += 1

    # Test 36: Absolute value at zero - d/dx(|x|) = 0 for x = 0 (using sign convention)
    x36 = Element(0)
    total += 1
    if test_case("Abs zero: f(x) = |x|",
                 lambda: abs(x36), [x36], [0]):
        passed += 1

    # Test 37: Abs composition - d/dx(|x²-4|) at x=3
    x37 = Element(3)
    total += 1
    # x² - 4 = 9 - 4 = 5 > 0, so |5| derivative is +1
    # d/dx(x² - 4) = 2x = 6
    # df/dx = sign(5) * 6 = 6
    if test_case("Abs composition: f(x) = |x² - 4| at x=3",
                 lambda: abs(x37 ** 2 - 4), [x37], [6]):
        passed += 1

    # Test 38: Abs composition with negative interior - d/dx(|x²-4|) at x=1
    x38 = Element(1)
    total += 1
    # x² - 4 = 1 - 4 = -3 < 0, so |-3| derivative is -1
    # d/dx(x² - 4) = 2x = 2
    # df/dx = sign(-3) * 2 = -2
    if test_case("Abs composition negative interior: f(x) = |x² - 4| at x=1",
                 lambda: abs(x38 ** 2 - 4), [x38], [-2]):
        passed += 1

    # Test 39: Nested abs - d/dx(||x||) = d/dx(|x|)
    x39 = Element(-5)
    total += 1
    if test_case("Nested abs: f(x) = ||x||",
                 lambda: abs(abs(x39)), [x39], [-1]):
        passed += 1

    # Test 40: Abs in product - d/dx(x * |x|)
    x40 = Element(2)
    total += 1
    # f(x) = x * |x| = x * x = x² (for x > 0)
    # df/dx = 2x = 4
    if test_case("Abs product positive: f(x) = x * |x|",
                 lambda: x40 * abs(x40), [x40], [4]):
        passed += 1

    # Test 41: Abs in product with negative - d/dx(x * |x|)
    x41 = Element(-2)
    total += 1
    # f(x) = x * |x| = x * (-x) = -x² (for x < 0)
    # df/dx = -2x = -2(-2) = 4
    if test_case("Abs product negative: f(x) = x * |x|",
                 lambda: x41 * abs(x41), [x41], [4]):
        passed += 1

    # UNARY NEGATION TESTS
    print(f"\n{'=' * 20} UNARY NEGATION TESTS {'=' * 20}")

    # Test 42: Simple negation - d/dx(-x) = -1
    x42 = Element(5)
    total += 1
    if test_case("Negation: f(x) = -x",
                 lambda: -x42, [x42], [-1]):
        passed += 1

    # Test 43: Negation of negative - d/dx(-x) = -1 even for negative x
    x43 = Element(-3)
    total += 1
    if test_case("Negation of negative: f(x) = -x where x < 0",
                 lambda: -x43, [x43], [-1]):
        passed += 1

    # Test 44: Negation composition - d/dx(-(x²)) = -2x
    x44 = Element(3)
    total += 1
    if test_case("Negation composition: f(x) = -(x²)",
                 lambda: -(x44 ** 2), [x44], [-6]):
        passed += 1

    # Test 45: Double negation - d/dx(-(-x)) = 1
    x45 = Element(4)
    total += 1
    if test_case("Double negation: f(x) = -(-x)",
                 lambda: -(-x45), [x45], [1]):
        passed += 1

    # Test 46: Negation in expression - d/dx(-x + x²) = -1 + 2x
    x46 = Element(3)
    total += 1
    # df/dx = -1 + 2(3) = 5
    if test_case("Negation in expression: f(x) = -x + x²",
                 lambda: -x46 + x46 ** 2, [x46], [5]):
        passed += 1

    # REVERSE OPERATION TESTS
    print(f"\n{'=' * 20} REVERSE OPERATION TESTS {'=' * 20}")

    # Test 47: Reverse addition - d/dx(5 + x) = 1
    x47 = Element(3)
    total += 1
    if test_case("Reverse add: f(x) = 5 + x",
                 lambda: 5 + x47, [x47], [1]):
        passed += 1

    # Test 48: Reverse subtraction - d/dx(10 - x) = -1
    x48 = Element(3)
    total += 1
    if test_case("Reverse subtract: f(x) = 10 - x",
                 lambda: 10 - x48, [x48], [-1]):
        passed += 1

    # Test 49: Reverse multiplication - d/dx(5 * x) = 5
    x49 = Element(4)
    total += 1
    if test_case("Reverse multiply: f(x) = 5 * x",
                 lambda: 5 * x49, [x49], [5]):
        passed += 1

    # Test 50: Reverse division - d/dx(12 / x) = -12/x²
    x50 = Element(2)
    total += 1
    # df/dx = -12/4 = -3
    if test_case("Reverse divide: f(x) = 12 / x",
                 lambda: 12 / x50, [x50], [-3]):
        passed += 1

    # Test 51: Reverse power - d/dx(2^x) = 2^x * ln(2)
    x51 = Element(3)
    total += 1
    # 2^3 = 8, df/dx = 8 * ln(2) ≈ 5.5452
    expected_grad = 8 * np.log(2)
    if test_case("Reverse power: f(x) = 2^x",
                 lambda: 2 ** x51, [x51], [expected_grad]):
        passed += 1

    # Test 52: Mixed forward and reverse - d/dx(x * 3 + 5 * x) = 3 + 5 = 8
    x52 = Element(2)
    total += 1
    if test_case("Mixed operations: f(x) = x * 3 + 5 * x",
                 lambda: x52 * 3 + 5 * x52, [x52], [8]):
        passed += 1

    # RESET METHOD TESTS
    print(f"\n{'=' * 20} RESET METHOD TESTS {'=' * 20}")

    # Test 53: Reset clears gradients
    x53 = Element(3)
    y53 = x53 ** 2
    y53.backward()
    stored_grad = x53._grad.item()
    x53.reset()
    total += 1
    # After reset, gradient should be 0
    passed_test = (x53._grad.item() == 0 and stored_grad == 6)
    print(f"\n=== Reset clears gradients ===")
    print(f"  Gradient before reset: {stored_grad:.6f} (expected 6.000000) {'✓' if stored_grad == 6 else '✗'}")
    print(f"  Gradient after reset: {x53._grad.item():.6f} (expected 0.000000) {'✓' if x53._grad.item() == 0 else '✗'}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 54: Multiple backward calls without reset accumulate
    x54 = Element(2)
    y54 = x54 ** 2
    y54.backward()
    first_grad = x54._grad.item()
    y54.backward()  # Second call without reset
    second_grad = x54._grad.item()
    total += 1
    # First: grad = 4, Second: grad = 4 + 4 = 8 (accumulation)
    passed_test = (first_grad == 4 and second_grad == 8)
    print(f"\n=== Multiple backward without reset ===")
    print(f"  Gradient after 1st backward: {first_grad:.6f} (expected 4.000000) {'✓' if first_grad == 4 else '✗'}")
    print(f"  Gradient after 2nd backward: {second_grad:.6f} (expected 8.000000) {'✓' if second_grad == 8 else '✗'}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 55: Reset propagates through graph
    x55, y55 = Element(2), Element(3)
    z55 = x55 * y55
    result55 = z55 ** 2
    result55.backward()
    grad_x_before = x55._grad.item()
    grad_y_before = y55._grad.item()
    grad_z_before = z55._grad.item()
    result55.reset()
    total += 1
    # All gradients should be 0 after reset
    passed_test = (x55._grad.item() == 0 and y55._grad.item() == 0 and z55._grad.item() == 0 and result55._grad.item() == 0)
    print(f"\n=== Reset propagates through graph ===")
    print(f"  x grad before: {grad_x_before:.6f}, after: {x55._grad.item():.6f} {'✓' if x55._grad.item() == 0 else '✗'}")
    print(f"  y grad before: {grad_y_before:.6f}, after: {y55._grad.item():.6f} {'✓' if y55._grad.item() == 0 else '✗'}")
    print(f"  z grad before: {grad_z_before:.6f}, after: {z55._grad.item():.6f} {'✓' if z55._grad.item() == 0 else '✗'}")
    print(f"  result grad before: 1.000000, after: {result55._grad.item():.6f} {'✓' if result55._grad.item() == 0 else '✗'}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 56: Backward after reset gives correct gradients
    x56 = Element(4)
    y56 = x56 ** 2
    y56.backward()
    x56.reset()
    # Now compute again with different expression
    z56 = x56 ** 3
    z56.backward()
    total += 1
    # df/dx = 3x² = 3(16) = 48
    passed_test = abs(x56._grad.item() - 48) < 1e-6
    print(f"\n=== Backward after reset ===")
    print(f"  Gradient: {x56._grad.item():.6f} (expected 48.000000) {'✓' if passed_test else '✗'}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # VECTOR OPERATION TESTS
    print(f"\n{'=' * 20} VECTOR OPERATION TESTS {'=' * 20}")

    # Test 57: Vector addition - element-wise
    x57 = Element([[1, 2], [3, 4]])
    y57 = Element([[5, 6], [7, 8]])
    z57 = x57 + y57
    # Create a scalar result by doing element-wise operations
    result57 = Element(1) * z57._value[0, 0] + Element(1) * z57._value[0, 1] + Element(1) * z57._value[1, 0] + Element(1) * z57._value[1, 1]
    # Manually construct gradient flow
    x57._grad = np.array([[1, 1], [1, 1]])
    y57._grad = np.array([[1, 1], [1, 1]])
    total += 1
    # z = [[6, 8], [10, 12]]
    # For addition: dL/dx = dL/dz = [[1,1],[1,1]]
    expected_val = np.array([[6., 8.], [10., 12.]])
    passed_test = np.allclose(z57._value, expected_val)
    print(f"\n=== Vector addition (forward pass) ===")
    print(f"  z value:\n{z57._value}")
    print(f"  Expected:\n{expected_val}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 58: Vector multiplication (element-wise) with backprop
    x58 = Element([[2, 3]])
    y58 = Element([[4, 5]])
    z58 = x58 * y58
    # Get single element for backward
    result58 = z58._value[0, 0] + z58._value[0, 1]
    result58_elem = Element(result58)
    result58_elem._left = z58
    result58_elem._grad = np.array([[1.]])
    # Manually do backward
    z58._grad = np.array([[1., 1.]])
    z58._grad_fn()
    total += 1
    # z = [[8, 15]], sum = 23
    # dL/dx = y * dL/dz = [[4, 5]]
    # dL/dy = x * dL/dz = [[2, 3]]
    expected_x = np.array([[4., 5.]])
    expected_y = np.array([[2., 3.]])
    passed_test = np.allclose(x58._grad, expected_x) and np.allclose(y58._grad, expected_y)
    print(f"\n=== Vector element-wise multiplication ===")
    print(f"  x grad:\n{x58._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y58._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 59: Matrix multiplication
    x59 = Element([[1, 2], [3, 4]])
    y59 = Element([[5, 6], [7, 8]])
    z59 = x59 @ y59
    # Manually propagate gradients
    z59._grad = np.array([[1., 1.], [1., 1.]])
    z59._grad_fn()
    total += 1
    # z = [[19, 22], [43, 50]]
    # dL/dx = dL/dz @ y.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    # dL/dy = x.T @ dL/dz = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    expected_x = np.array([[11., 15.], [11., 15.]])
    expected_y = np.array([[4., 4.], [6., 6.]])
    passed_test = np.allclose(x59._grad, expected_x) and np.allclose(y59._grad, expected_y)
    print(f"\n=== Matrix multiplication ===")
    print(f"  x grad:\n{x59._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y59._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 60: Transpose operation
    x60 = Element([[1, 2, 3], [4, 5, 6]])
    z60 = x60.T
    # Manually propagate gradients
    z60._grad = np.array([[1., 1.], [1., 1.], [1., 1.]])
    z60._grad_fn()
    total += 1
    # z = [[1,4],[2,5],[3,6]]
    # dL/dx = dL/dz.T = [[1,1,1],[1,1,1]]
    expected_x = np.array([[1., 1., 1.], [1., 1., 1.]])
    passed_test = np.allclose(x60._grad, expected_x)
    print(f"\n=== Transpose operation ===")
    print(f"  x grad:\n{x60._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 61: Vector subtraction
    x61 = Element([[5, 6], [7, 8]])
    y61 = Element([[1, 2], [3, 4]])
    z61 = x61 - y61
    # Manually propagate gradients
    z61._grad = np.array([[1., 1.], [1., 1.]])
    z61._grad_fn()
    total += 1
    # z = [[4, 4], [4, 4]]
    # dL/dx = [[1, 1], [1, 1]], dL/dy = -[[1, 1], [1, 1]]
    expected_x = np.array([[1., 1.], [1., 1.]])
    expected_y = -np.array([[1., 1.], [1., 1.]])
    passed_test = np.allclose(x61._grad, expected_x) and np.allclose(y61._grad, expected_y)
    print(f"\n=== Vector subtraction ===")
    print(f"  x grad:\n{x61._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y61._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 62: Vector division
    x62 = Element([[6, 8], [10, 12]])
    y62 = Element([[2, 4], [5, 6]])
    z62 = x62 / y62
    # Manually propagate gradients
    z62._grad = np.array([[1., 1.], [1., 1.]])
    z62._grad_fn()
    total += 1
    # z = [[3, 2], [2, 2]]
    # dL/dx = 1/y = [[0.5, 0.25], [0.2, 0.1667]]
    # dL/dy = -x/y² = [[-1.5, -0.5], [-0.4, -0.3333]]
    expected_x = 1.0 / np.array([[2., 4.], [5., 6.]])
    expected_y = -np.array([[6., 8.], [10., 12.]]) / np.array([[2., 4.], [5., 6.]]) ** 2
    passed_test = np.allclose(x62._grad, expected_x) and np.allclose(y62._grad, expected_y)
    print(f"\n=== Vector division ===")
    print(f"  x grad:\n{x62._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y62._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 63: Vector power (element-wise)
    x63 = Element([[2, 3], [4, 5]])
    y63 = Element([[2, 2], [2, 2]])
    z63 = x63 ** y63
    # Manually propagate gradients
    z63._grad = np.array([[1., 1.], [1., 1.]])
    z63._grad_fn()
    total += 1
    # z = [[4, 9], [16, 25]]
    # dL/dx = y * x^(y-1) = [[2*2^1, 2*3^1], [2*4^1, 2*5^1]] = [[4, 6], [8, 10]]
    expected_x = 2 * np.array([[2., 3.], [4., 5.]])
    passed_test = np.allclose(x63._grad, expected_x)
    print(f"\n=== Vector power (element-wise) ===")
    print(f"  x grad:\n{x63._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 64: Element-wise multiplication with constant
    x64 = Element([[1, 2], [3, 4]])
    scalar = Element([[3, 3], [3, 3]])
    z64 = x64 * scalar
    # Manually propagate gradients
    z64._grad = np.array([[1., 1.], [1., 1.]])
    z64._grad_fn()
    total += 1
    # z = [[3, 6], [9, 12]]
    # dL/dx = scalar * dL/dz = [[3,3],[3,3]]
    expected_x = np.array([[3., 3.], [3., 3.]])
    passed_test = np.allclose(x64._grad, expected_x)
    print(f"\n=== Element-wise multiplication with constant ===")
    print(f"  x grad:\n{x64._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 65: Vector absolute value
    x65 = Element([[-2, 3], [4, -5]])
    z65 = abs(x65)
    # Manually propagate gradients
    z65._grad = np.array([[1., 1.], [1., 1.]])
    z65._grad_fn()
    total += 1
    # z = [[2, 3], [4, 5]]
    # dL/dx = sign(x) = [[-1, 1], [1, -1]]
    expected_x = np.sign(np.array([[-2., 3.], [4., -5.]]))
    passed_test = np.allclose(x65._grad, expected_x)
    print(f"\n=== Vector absolute value ===")
    print(f"  x grad:\n{x65._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 66: Vector negation (using subtraction instead)
    x66 = Element([[1, 2], [3, 4]])
    zero = Element([[0, 0], [0, 0]])
    z66 = zero - x66
    # Manually propagate gradients
    z66._grad = np.array([[1., 1.], [1., 1.]])
    z66._grad_fn()
    total += 1
    # z = [[-1, -2], [-3, -4]]
    # dL/dx = -dL/dz = -[[1,1],[1,1]] = [[-1,-1],[-1,-1]]
    expected_x = np.array([[-1., -1.], [-1., -1.]])
    passed_test = np.allclose(x66._grad, expected_x)
    print(f"\n=== Vector negation ===")
    print(f"  x grad:\n{x66._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 67: Rectangular matrix multiplication
    x67 = Element([[1, 2, 3], [4, 5, 6]])  # 2x3
    y67 = Element([[1, 2], [3, 4], [5, 6]])  # 3x2
    z67 = x67 @ y67  # 2x2
    # Manually propagate gradients
    z67._grad = np.array([[1., 1.], [1., 1.]])
    z67._grad_fn()
    total += 1
    # z = [[22, 28], [49, 64]]
    # dL/dx = dL/dz @ y.T = [[1,1],[1,1]] @ [[1,3,5],[2,4,6]] = [[3,7,11],[3,7,11]]
    # dL/dy = x.T @ dL/dz = [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
    expected_x = np.array([[3., 7., 11.], [3., 7., 11.]])
    expected_y = np.array([[5., 5.], [7., 7.], [9., 9.]])
    passed_test = np.allclose(x67._grad, expected_x) and np.allclose(y67._grad, expected_y)
    print(f"\n=== Rectangular matrix multiplication ===")
    print(f"  x grad:\n{x67._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y67._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 68: Vector with sigmoid
    x68 = Element([[0, 1], [-1, 2]])
    z68 = sigmoid(x68)
    # Manually propagate gradients
    z68._grad = np.array([[1., 1.], [1., 1.]])
    z68._grad_fn()
    total += 1
    # For each element: grad = sigmoid(x) * (1 - sigmoid(x))
    sig_vals = 1.0 / (1.0 + np.exp(-np.array([[0., 1.], [-1., 2.]])))
    expected_x = sig_vals * (1 - sig_vals)
    passed_test = np.allclose(x68._grad, expected_x)
    print(f"\n=== Vector with sigmoid ===")
    print(f"  x grad:\n{x68._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    # Test 69: Chain of matrix operations (matmul + transpose)
    x69 = Element([[1, 2], [3, 4]])
    y69 = Element([[2, 0], [0, 2]])
    z69 = x69 @ y69
    w69 = z69.T
    # Manually propagate gradients backwards
    w69._grad = np.array([[1., 1.], [1., 1.]])
    w69._grad_fn()  # w = z.T, so dL/dz = dL/dw.T
    z69._grad_fn()  # z = x @ y
    total += 1
    # z = [[2, 4], [6, 8]], z.T = [[2, 6], [4, 8]]
    # dL/dz = dL/dw.T = [[1,1],[1,1]]
    # dL/dx = dL/dz @ y.T = [[1,1],[1,1]] @ [[2,0],[0,2]] = [[2,2],[2,2]]
    # dL/dy = x.T @ dL/dz = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    expected_x = np.array([[2., 2.], [2., 2.]])
    expected_y = np.array([[4., 4.], [6., 6.]])
    passed_test = np.allclose(x69._grad, expected_x) and np.allclose(y69._grad, expected_y)
    print(f"\n=== Chain: matmul + transpose ===")
    print(f"  x grad:\n{x69._grad}")
    print(f"  Expected:\n{expected_x}")
    print(f"  y grad:\n{y69._grad}")
    print(f"  Expected:\n{expected_y}")
    print(f"  Result: {'PASS' if passed_test else 'FAIL'}")
    if passed_test:
        passed += 1

    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"Success rate: {100 * passed / total:.1f}%")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed - check implementation")

if __name__ == "__main__":
    run_tests()