from element import Element

def announce(title, n_repeat=10):
    print("=" * n_repeat, title, "=" * n_repeat)

if __name__ == "__main__":
    x = Element(2)
    y = Element(3)

    print(x._grad, y._grad)

    z = x * y
    a = z ** 2

    print(x._grad, y._grad)

    a.backward()
    print(x._grad, y._grad)
    
    a.reset()
    announce("Reset")

    print(x._grad, y._grad)
    a = ((x * y) ** 2)
    print(x._grad, y._grad)

    a.backward()
    print(x._grad, y._grad)

    announce("c^2 + c, c:=4")
    c = Element(4)
    q = c ** 2 + c

    q._grad = 1
    q.backward()

    print(c._grad)
