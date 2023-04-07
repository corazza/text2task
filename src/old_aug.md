
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
