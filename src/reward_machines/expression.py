class Expression:
    def __init__(self):
        return

    def test(self, input_symbol: frozenset[str]) -> bool:
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class Var(Expression):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.symbol in input_symbol

    def __str__(self) -> str:
        return self.symbol


class Not(Expression):
    def __init__(self, expr: Expression):
        super().__init__()
        self.child = expr

    def test(self, input_symbol: frozenset[str]) -> bool:
        return not self.child.test(input_symbol)

    def __str__(self):
        return f'!({self.child})'


class Or(Expression):
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left = left
        self.right = right

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.left.test(input_symbol) or self.right.test(input_symbol)

    def __str__(self):
        return f'({self.left} | {self.right})'


class And(Expression):
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left = left
        self.right = right

    def test(self, input_symbol: frozenset[str]) -> bool:
        return self.left.test(input_symbol) and self.right.test(input_symbol)

    def __str__(self):
        return f'({self.left} & {self.right})'


def compile(expr: Expression) -> set[frozenset[str]]:
    raise NotImplementedError()
