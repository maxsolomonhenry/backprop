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
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"Success rate: {100 * passed / total:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed - check implementation")

if __name__ == "__main__":
    run_tests()