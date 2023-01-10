import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value={self.data},Grad:{self.grad}"

    def __radd__(self, other):
        return self + other

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = backward
        return output

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward
        return output

    def __pow__(self, other) -> "Value":
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data**other)

        def backward():
            self.grad += other * (self.data ** (other - 1)) * output.grad

        output._backward = backward
        return output

    def __rmul__(self, other) -> "Value":  # other * self fallback
        return self * other

    def __truediv__(self, other) -> "Value":  # self / other
        return self * other**-1

    def __neg__(self) -> "Value":  # -self
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __gt__(self, other) -> bool:
        return self.data > other.data

    def __lt__(self, other) -> bool:
        return self.data < other.data

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * output.grad

        output._backward = backward
        return output

    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self,), "exp")

        def backward():
            self.grad = output.data * output.grad

        output._backward = backward
        return output

    def log(self):
        x = self.data
        output = Value(math.log(x), (self,), "log")

        def backward():
            self.grad = 1 / x

        output._backward = backward
        return output

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
