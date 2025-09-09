#!/usr/bin/env python3

from element import Element

def test_case(name, computation, variables, expected_grads, tolerance=1e-6):
    """Test a single case and report results"""
    print(f"\n=== {name} ===")
    
    # Reset all variables
    for var in variables:
        var._grad = 0
    
    # Run computation and backward pass
    result = computation()
    result.backward()
    
    # Check results
    all_passed = True
    for i, (var, expected) in enumerate(zip(variables, expected_grads)):
        actual = var._grad
        passed = abs(actual - expected) < tolerance
        status = "✓" if passed else "✗"
        print(f"  Var {i+1}: expected={expected:.6f}, actual={actual:.6f} {status}")
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
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"Success rate: {100 * passed / total:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed - check implementation")

if __name__ == "__main__":
    run_tests()