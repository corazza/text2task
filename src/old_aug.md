
def add_avoidance(examples: list[Example]) -> list[Example]:
    new_examples: list[Example] = []
    for example in examples:
        eligible_classes = ['NO_THE', 'OBJECT_A',
                            'AVOID', 'STEP_ON_A', 'FALL_IN_A', 'HIT_A']
        pvar_class = random_from(eligible_classes)
        if np.random.random() < ADD_AVOIDANCE_P/2.0:
            new_examples.append(with_avoidance(pvar_class, example))
    print(f'avoidance added {len(new_examples)} new examples')
    return examples + new_examples

def add_concat(examples: list[Example]) -> list[Example]:
    num_eligible = len(
        [x for x in examples if x.average_desc_length() <= DESC_LENGTH_LIMIT])
    add_p = ADD_CONCAT_P / num_eligible
    print(f'eligible={num_eligible}/{len(examples)}')
    new_examples: list[Example] = []
    for i, example1 in enumerate(examples):
        for j, example2 in enumerate(examples):
            if i == j:
                continue
            if example1.average_desc_length() > DESC_LENGTH_LIMIT or example2.average_desc_length() > DESC_LENGTH_LIMIT:
                continue
            if len(example1.example_rewrites) == 0 or len(example2.example_rewrites) == 0:
                continue  # TODO fix in with_concat
            if np.random.random() < add_p:
                new_examples.append(with_concat(example1, example2))
    print(f'concat added {len(new_examples)} new examples')
    return examples + new_examples

def with_avoidance(pvar_class: str, example: Example) -> Example:
    variable = '$Z'
    desc_before: dict[str, list[str]] = {
        'NO_THE': [],
        'OBJECT_A': [],
        'AVOID': [],
        'STEP_ON_A': [],
        'FALL_IN_A': [],
        'HIT_A': [],
    }
    desc_after: dict[str, list[str]] = {
        'NO_THE': [],
        'OBJECT_A': [],
        'AVOID': [],
        'STEP_ON_A': [],
        'FALL_IN_A': [],
        'HIT_A': [],
    }

    desc_before['NO_THE'].append(f'Avoid {variable}. ')
    desc_before['NO_THE'].append(
        f'Do the following task, but avoid {variable}: ')
    desc_after['NO_THE'].append(f'. Avoid {variable}.')
    desc_after['NO_THE'].append(f', and avoid {variable}.')
    desc_after['NO_THE'].append(f'. Don\'t get near {variable}.')

    desc_before['OBJECT_A'].append(f'Avoid {variable}s. ')
    desc_before['OBJECT_A'].append(
        f'Do the following task, but avoid {variable}s: ')
    desc_after['OBJECT_A'].append(f'. Avoid {variable}s.')
    desc_after['OBJECT_A'].append(f', and avoid {variable}s')
    desc_after['OBJECT_A'].append(f'. Don\'t get near {variable}s')

    desc_before['AVOID'].append(f'Avoid {variable}s. ')
    desc_before['AVOID'].append(
        f'Do the following task, but avoid {variable}s: ')
    desc_after['AVOID'].append(f'. Avoid {variable}s')
    desc_after['AVOID'].append(f', and avoid {variable}s.')
    desc_after['AVOID'].append(f'. Don\'t get near {variable}s')

    desc_before['STEP_ON_A'].append(f'Don\'t step on {variable}s. ')
    desc_before['STEP_ON_A'].append(
        f'Do the following task, but don\'t step on {variable}s: ')
    desc_after['STEP_ON_A'].append(f', and never step on {variable}s')
    desc_after['STEP_ON_A'].append(f'. Don\'t step on {variable}s.')

    desc_before['FALL_IN_A'].append(f'Don\'t fall in a {variable}. ')
    desc_before['FALL_IN_A'].append(
        f'Do the following task, but don\'t fall in a {variable}: ')
    desc_after['FALL_IN_A'].append(f', and never fall in a {variable}.')
    desc_after['FALL_IN_A'].append(f'. Don\'t fall in a {variable}s')

    desc_before['HIT_A'].append(f'Don\'t hit a {variable}. ')
    desc_before['HIT_A'].append(
        f'Do the following task, but don\'t hit a {variable}: ')
    desc_after['HIT_A'].append(f', and never hit a {variable}')
    desc_after['HIT_A'].append(f'. Don\'t hit a {variable}s.')

    src_after: list[str] = [
        f'(!{variable})*',  # (!$B)*
        f'((.)* > {variable} > (.)*)~',  # ((.)* > $B > (.)*)~
    ]

    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = [rewrite_list + [(variable, pvar_class)] for rewrite_list in example.example_rewrites]
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example.representative()}_ADDED_AVOID'

    for (reward, pvars) in example.runs:
        new_runs.append((reward, pvars))
        new_runs.append((0, [frozenset({variable})] + pvars))

    for desc in example.descs:
        if np.random.random() < AUGMENT_PREFER_BEFORE:
            before: str = random_from(desc_before[pvar_class])
            new_desc = f'{before}{desc}'
        else:
            after: str = random_from(desc_after[pvar_class])
            new_desc = f'{desc}{after}'
        new_desc = clean_desc(new_desc)
        new_descs.append(new_desc)

    for src in example.srcs:
        for after in src_after:
            new_src = f'({src})&{after}'
            new_srcs.append(new_src)

    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)

def with_concat(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_CONCAT_{example2.representative()}'

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'({src1}) > ({src2})'
            new_srcs.append(new_src)

    if VALIDATE_AB_AUGMENTS:
        concat_rm = compiler_interface.compile(new_srcs[0])
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])
        for (_reward1, _trace1) in example1.runs:
            reward1 = example1_rm(*_trace1)
            first_positive = next(
                (i for i, x in enumerate(reward1) if x > 0), None)
            for (_reward2, trace2) in example2.runs:
                reward2 = example2_rm(*trace2)
                if first_positive == None:
                    trace1 = _trace1
                    reward = 0
                else:
                    trace1 = _trace1[:first_positive+1]
                    reward = sum(reward2)
                trace = map_trace('Q', trace1) + map_trace('P', trace2)
                reward = sum(concat_rm(*trace))  # TODO this is wrong
                new_runs.append((reward, trace))

    for desc1_or in example1.descs:
        if '.' in desc1_or:
            continue
        for desc2 in example2.descs:
            if '.' in desc2:
                continue
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            # adding new descs
            if 'first' not in desc1.lower() and 'second' not in desc2.lower():
                new_descs.append(f'First: {desc1}. Second: {desc2}')
                new_descs.append(
                    f'There are two tasks. First {desc1}. Second {desc2}.')
            if 'first' not in desc1.lower():
                new_descs.append(f'First: {desc1}. Then: {desc2}')
                new_descs.append(
                    f'There are two tasks. First {desc1}. Second {desc2}.')
            new_descs.append(f'{desc1}. Then {desc2}')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)

def with_disjunct(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_DISJUNCT_{example2.representative()}'

    if VALIDATE_AB_AUGMENTS:
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])

        for (reward, trace) in example1.runs:
            reward1 = example1_rm(*trace)
            reward2 = example2_rm(*trace)
            new_runs.append(
                (sum(reward1) + sum(reward2), map_trace('Q', trace)))

        for (reward, trace) in example2.runs:
            reward1 = example1_rm(*trace)
            reward2 = example2_rm(*trace)
            new_runs.append(
                (sum(reward1) + sum(reward2), map_trace('P', trace)))

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'({src1}) | ({src2})'
            new_srcs.append(new_src)

    for desc1_or in example1.descs:
        if '.' in desc1_or:
            continue
        for desc2 in example2.descs:
            if '.' in desc2:
                continue
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            # adding new descs
            new_descs.append(
                f'There are two tasks. Either {desc1}, or {desc2}.')
            new_descs.append(
                f'Either {desc1}. Or, {desc2}.')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)

def merge_traces(trace1: list[frozenset[str]], trace2: list[frozenset[str]]) -> list[frozenset[str]]:
    trace: list[frozenset[str]] = []
    for vars1, vars2 in itertools.zip_longest(trace1, trace2, fillvalue=frozenset()):
        trace.append(vars1.union(vars2))
    return trace

def merge_rewards(rewards1: list[int], rewards2: list[int]):
    rewards: list[int] = []
    for r1, r2 in itertools.zip_longest(rewards1, rewards2, fillvalue=0):
        rewards.append(r1*r2)
    return rewards

def with_conjunct(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_CONJUNCT_{example2.representative()}'

    if VALIDATE_AB_AUGMENTS:
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])

        for (_reward1, trace1) in example1.runs:
            reward1 = example1_rm(*trace1)
            for (_reward2, trace2) in example2.runs:
                reward2 = example2_rm(*trace2)
                rewards = merge_rewards(reward1, reward2)
                reward = sum(rewards)
                trace_merged = merge_traces(trace1, trace2)
                new_runs.append((reward, trace_merged))

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            ast1 = compiler_interface.parse(src1)
            ast2 = compiler_interface.parse(src2)
            first_terminator = '' if ast1.repetative() else ' > (.)*'
            second_terminator = '' if ast2.repetative() else ' > (.)*'
            new_src = f'(({src1}){first_terminator}) & (({src2}){second_terminator})'
            new_srcs.append(new_src)

    for desc1_or in example1.descs:
        if '.' in desc1_or:
            continue
        for desc2 in example2.descs:
            if '.' in desc2:
                continue
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            # adding new descs
            new_descs.append(
                f'There are two tasks and you must do both at the same time. {desc1} (first task), and {desc2} (second task).')
            new_descs.append(
                f'{desc1}. And at the same time, {desc2}.')
            new_descs.append(
                f'{desc1}. And, {desc2}.')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)

def augmented_ab(examples: list[Example], num_abs) -> list[Tuple[Example, str, str]]:
    new_abs: list[Tuple[Example, str, str]] = []
    np.random.shuffle(examples)  # type: ignore
    eligible_pairs: list[Tuple[int, int]] = get_eligible_pairs(examples)

    num_concat: int = round(ADD_CONCAT_P * ADD_P * num_abs)
    assert len(eligible_pairs) >= num_concat
    print(f'adding concat... ({num_concat})')
    np.random.shuffle(eligible_pairs)
    for i, j in eligible_pairs[:num_concat]:
        example1 = examples[i]
        example2 = examples[j]
        example = with_concat(example1, example2)  # type: ignore
        new_abs.append(get_single_ab(example))

    num_disjunct: int = round(ADD_DISJUNCT_P * ADD_P * num_abs)
    print(f'adding disjunct... ({num_disjunct})')
    np.random.shuffle(eligible_pairs)
    for i, j in eligible_pairs[:num_disjunct]:
        example1 = examples[i]
        example2 = examples[j]
        example = with_disjunct(example1, example2)
        new_abs.append(get_single_ab(example))

    num_conjunct: int = round(ADD_CONJUNCT_P * ADD_P * num_abs)
    print(f'adding conjunct... ({num_conjunct})')
    np.random.shuffle(eligible_pairs)
    for i, j in eligible_pairs[:num_conjunct]:
        example1 = examples[i]
        example2 = examples[j]
        example = with_conjunct(example1, example2)
        new_abs.append(get_single_ab(example))

    num_avoidance: int = round(ADD_AVOIDANCE_P * ADD_P * num_abs)
    print(f'adding avoidance... ({num_avoidance})')
    eligible_classes = ['NO_THE', 'OBJECT_A',
                        'AVOID', 'STEP_ON_A', 'FALL_IN_A', 'HIT_A']

    for i in range(num_avoidance):
        example = random_from(examples)
        pvar_class = random_from(eligible_classes)
        new_example = with_avoidance(pvar_class, example)
        new_abs.append(get_single_ab(new_example))

    if VALIDATE_AB_AUGMENTS:
        validate_runs([e for e, a, b in new_abs])

    return new_abs
