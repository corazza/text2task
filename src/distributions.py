import numpy as np
from typing import Tuple


class DistBase:
    def sample_int(self) -> int:
        return int(self.sample())

    def sample(self) -> float:
        raise NotImplementedError()

    def sample_bool(self) -> bool:
        raise NotImplementedError()

    def sample_string(self, default: str) -> str:
        raise NotImplementedError()

    def sample_string_banned(self, banned: set[str], default: str) -> str:
        raise NotImplementedError()

    def __str__(self):
        return repr(self)


class ShiftedExpDist(DistBase):
    def __init__(self, start: float, avg: float):
        assert start < avg
        self.start = start
        self.avg = avg

    def __repr__(self):
        return f'ShiftedExp({self.start:.2f}, {self.avg:.2f})'

    def __str__(self):
        return repr(self)

    def sample(self) -> float:
        return self.start + np.round(np.random.exponential(self.avg - self.start))


class ShiftedLimitedExpDist(DistBase):
    def __init__(self, start: float, avg: float, limit: float):
        assert limit > avg
        self.limit = limit
        self.underlying = ShiftedExpDist(start, avg)

    def __repr__(self):
        return f'ShiftedLimitedExp({self.underlying.start:.2f}, {self.underlying.avg:.2f}, {self.limit:.2f})'

    def sample(self) -> float:
        while True:
            x = self.underlying.sample()
            if x <= self.limit:
                return x


class HistogramDist(DistBase):
    def __init__(self, values: list[int | float]):
        self.values = values
        self.num_samples = len(values)
        self.avg = float(sum(values)) / self.num_samples

    def __repr__(self):
        return f'HistogramDist(avg={self.avg:.2f})'

    def sample(self) -> float:
        chosen = np.random.randint(self.num_samples)
        return self.values[chosen]


class BinaryDist(DistBase):
    def __init__(self, p):
        assert 0 < p and p < 1
        self.p = p

    def __repr__(self):
        return f'Binary({self.p:.2f})'

    def sample_bool(self) -> bool:
        return np.random.random() < self.p


class ChoiceDist(DistBase):
    def __init__(self, options: dict[str, int]):
        self.original_options = options
        self.options, self.collected = ChoiceDist.generate_options(options)

    @staticmethod
    def generate_options(options: dict[str, int]) -> Tuple[dict[str, int], int]:
        r = {}
        collected = 0
        for option in options:
            if options[option] > 0:
                r[option] = options[option] + collected
                collected += options[option]
        return (r, collected)

    # @staticmethod
    # def sample_uniform(options: dict[str, int]):
    #     keys = list(options.keys())
    #     choice = np.random.randint(len(keys))
    #     return keys[choice]

    @staticmethod
    def sample_from_options(options: dict[str, int], collected: int, default: str) -> str:
        if collected == 0:
            return default
        chosen = np.random.randint(1, collected+1)
        for option in options:
            if chosen <= options[option]:
                return option
        raise ValueError()

    def sample_string(self, default: str) -> str:
        return ChoiceDist.sample_from_options(self.options, self.collected, default)

    def sample_string_banned(self, banned: set[str], default: str) -> str:
        options = {option: num for option,
                   num in self.original_options.items() if option not in banned}
        (new_options, collected) = ChoiceDist.generate_options(options)
        r = ChoiceDist.sample_from_options(new_options, collected, default)
        return r

    def __repr__(self):
        return f'ChoiceDist({self.original_options})'
