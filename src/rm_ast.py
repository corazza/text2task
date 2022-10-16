class Expression:
    def __init__(self):
        return

    def appears(self) -> frozenset[str]:
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class Var(Expression):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def appears(self) -> frozenset[str]:
        return frozenset(self.symbol)

    def __str__(self) -> str:
        return self.symbol


class Or(Expression):
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left = left
        self.right = right

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def __str__(self):
        return f'({self.left} | {self.right})'


class Then(Expression):
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left = left
        self.right = right

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def __str__(self):
        return f'({self.left} -> {self.right})'


class Repeat(Expression):
    def __init__(self, child: Expression):
        super().__init__()
        self.child = child

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def __str__(self):
        return f'({self.child})*'
