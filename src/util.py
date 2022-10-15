from typing import Iterator


def select(a: list, b: int) -> frozenset:
    r = set()
    i = 0
    while b > 0:
        (b, selected) = divmod(b, 2)
        if selected:
            r.add(a[i])
        i += 1
    return frozenset(r)


def powerset(a: frozenset) -> Iterator[frozenset]:
    b = list(a)
    for i in range(2**len(a)):
        yield select(b, i)
