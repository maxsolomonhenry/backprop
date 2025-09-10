from element import Element

def announce(title, n_repeat=10):
    print("=" * n_repeat, title, "=" * n_repeat)

if __name__ == "__main__":
    announce("Expected gradient.")
    x = Element(2)

    z = (x * x) * (x * x)
    z.backward()
    print(x)

    announce("Bug: Gradient is different here.")
    x1 = Element(2)
    y1 = x1 * x1 # y1 = x1^2
    z1 = y1 * y1 # z1 = (x1^2)^2
    z1.backward()
    print(x1)