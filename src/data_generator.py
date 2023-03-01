from pathlib import Path
from typing import Tuple
import numpy as np
import IPython

import describe
import describe_patterns
import rm_generator
import expr_printer
import compiler_interface
import rm_ast


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


def generate_synthetic(props_path: str | Path, var_path: str | Path, patterns_path: str | Path, dist_parameters: dict[str, DistBase], n: int) -> list[Tuple[str, str]]:
    var_describe_map = describe.load_var_describe_map(var_path)
    patterns = describe_patterns.load_patterns(patterns_path)

    exprs = rm_generator.generate_many(
        props_path, dist_parameters, n)

    prompts = []

    for i in range(n):
        desc = describe.describe(patterns, var_describe_map, exprs[i])
        prompts.append(
            (desc.lower(), expr_printer.expr_to_str(exprs[i], randomize=True)))

    return prompts


def analyze_stats(expr: rm_ast.RMExpr) -> dict[str, int | list[int]]:
    max_level = describe.compute_max_level(expr)
    dist = {  # TODO write a class
        'nums_children': [],  # num. children per parent node
        'nums_props': [],  # num. props per leaf
        'num_negate': 0,  # total number of negated pvars
        'num_non_negate': 0,  # total number of non-negated pvars
        'max_level': max_level,  # max level per expr
        'num_repeats': 0,  # num. repeats per expr
        'num_plus': 0,  # num. repeats per expr
        'num_repeat': 0,  # num. repeats per expr
        'num_then': 0,  # num. repeats per expr
        'num_or': 0,  # num. repeats per expr
        'num_var': 0,  # num. repeats per expr
        'complexity': 0
    }
    to_visit = [expr]
    while len(to_visit) != 0:
        visiting = to_visit.pop(0)
        dist['complexity'] += 1
        if isinstance(visiting, rm_ast.Vars):
            for symbol in visiting.symbols:
                if '!' in symbol:
                    dist['num_negate'] += 1
                else:
                    dist['num_non_negate'] += 1
            dist['nums_props'].append(len(visiting.symbols))
            dist['num_var'] += 1
        elif isinstance(visiting, rm_ast.Or) or isinstance(visiting, rm_ast.Then):
            dist['nums_children'].append(len(visiting.exprs))
            to_visit.extend(visiting.exprs)
            if isinstance(visiting, rm_ast.Or):
                dist['num_or'] += 1
            else:
                dist['num_then'] += 1
        elif isinstance(visiting, rm_ast.Repeat) or isinstance(visiting, rm_ast.Plus):
            dist['num_repeats'] += 1
            to_visit.append(visiting.child)
            if isinstance(visiting, rm_ast.Repeat):
                dist['num_repeat'] += 1
            else:
                dist['num_plus'] += 1
        else:
            raise ValueError('unknown RM Expr AST type')
    return dist


def analyze_dist(prompts: list[Tuple[str, str]]) -> dict[str, DistBase]:
    compiled = list(
        map(lambda x: compiler_interface.parse(x[1]), prompts))
    stats = list(map(analyze_stats, compiled))

    children = 0
    total_children = 0

    props = 0
    total_props = 0

    total_negate = 0
    total_non_negate = 0

    # total_max_level = 0
    # total_repeats = 0
    total_complexity = 0

    complexities = list()
    max_levels = list()
    repeats = list()
    num_children = list()

    num_repeat = 0
    num_plus = 0
    num_then = 0
    num_or = 0
    num_var = 0

    # max_max_level = 0

    for stat in stats:
        num_children.extend(stat['nums_children'])  # type: ignore
        children += sum(stat['nums_children'])  # type: ignore
        total_children += len(stat['nums_children'])  # type: ignore
        props += sum(stat['nums_props'])  # type: ignore
        total_props += len(stat['nums_props'])  # type: ignore
        total_negate += stat['num_negate']  # type: ignore
        total_non_negate += stat['num_non_negate']  # type: ignore
        complexities.append(stat['complexity'])
        max_levels.append(stat['max_level'])
        repeats.append(stat['num_repeats'])

        num_repeat += stat['num_repeat']  # type: ignore
        num_plus += stat['num_plus']  # type: ignore
        num_then += stat['num_then']  # type: ignore
        num_or += stat['num_or']  # type: ignore
        num_var += stat['num_var']  # type: ignore

        # total_max_level += stat['max_level']  # type: ignore
        # total_repeats += stat['num_repeats']  # type: ignore
        total_complexity += stat['complexity']  # type: ignore

        # if stat['max_level'] > max_max_level:  # type: ignore
        #     max_max_level = stat['max_level']

    # assert isinstance(max_max_level, int)

    options = {
        'THEN': num_then,
        'OR': num_or,
        'PLUS': num_plus,
        'REPEAT': num_repeat,
        'VAR': num_var,
    }

    # TODO test invariance to generating from these stats
    num_samples = len(prompts)
    avg_children = children / total_children
    avg_props = props / total_props
    p_negate = total_negate / (total_negate + total_non_negate)
    # avg_max_level = total_max_level / num_samples
    # avg_repeats = total_repeats / num_samples
    avg_complexity = total_complexity / num_samples

    return {
        # defines distr. for # of children of nodes
        # 'children': ShiftedLimitedExpDist(2, avg_children, 4),
        'children': HistogramDist(num_children),
        # defines distr. for # of propvars in transitions
        'props': ShiftedLimitedExpDist(1, avg_props, 4),
        # probability to negate a propvar in transitions
        'negate': BinaryDist(p_negate),
        'max_level': HistogramDist(max_levels),
        'repeats': HistogramDist(repeats),
        'complexity': HistogramDist(complexities),
        'node': ChoiceDist(options)
    }
