from rm_util import powerset


class TExp:
    def __init__(self):
        return

    def test(self, input_symbol: frozenset[str]) -> bool:
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class Var(TExp):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.symbol in input_symbol

    def appears(self) -> frozenset[str]:
        return frozenset({self.symbol})

    def __str__(self) -> str:
        return self.symbol


class Not(TExp):
    def __init__(self, expr: TExp):
        super().__init__()
        self.child = expr

    def test(self, input_symbol: frozenset[str]) -> bool:
        return not self.child.test(input_symbol)

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def __str__(self):
        return f'!({self.child})'


class Or(TExp):
    def __init__(self, left: TExp, right: TExp):
        super().__init__()
        self.left = left
        self.right = right

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.left.test(input_symbol) or self.right.test(input_symbol)

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def __str__(self):
        return f'({self.left} | {self.right})'


class And(TExp):
    def __init__(self, left: TExp, right: TExp):
        super().__init__()
        self.left = left
        self.right = right

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.left.test(input_symbol) and self.right.test(input_symbol)

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def __str__(self):
        return f'({self.left} & {self.right})'


def compile(expr: TExp, appears=None) -> frozenset[frozenset[str]]:
    r = set()
    if appears == None:
        appears = expr.appears()
    for vars in powerset(appears):
        if expr.test(vars):
            r.add(vars)
    return frozenset(r)
