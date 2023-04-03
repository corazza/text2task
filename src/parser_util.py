from typing import Iterator, Optional, Tuple


def line_iter(path: str) -> Iterator[str]:
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#' or line[:2] == ' #':
                continue
            yield line.strip()
    yield ''
