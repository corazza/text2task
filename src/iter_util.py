import itertools
from typing import Iterator, Any


def with_first(t: Any, lex: Iterator[Any]) -> Iterator[Any]:
    return itertools.chain([t], lex)
