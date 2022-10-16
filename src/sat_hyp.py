from pysat.solvers import Glucose4
import itertools
import IPython

INITIAL_STATE = 0
TERMINAL_STATE = -1


def sample_language(X):
    """
    Returns the set of all values for true_props strings for a given counter-example set X
    E.g. the sample language for {(("b", "ab"), (0.0, 1.0)), (("ab", "a", "f"), (0.0, 0.0, 1.0))}
    is {"", "f", "b", "a", "ab"}.
    """
    language = set()
    language.add("")  # always contains empty string
    for (labels, _rewards) in X:
        language.update(labels)
    return language


def dnf_for_empty(language):
    """
    Returns the "neutral" CNF for a given sample language corresponding
    to no events being true
    Convenience method. Works on the result of sample_language(X).
    Semantically equivalent to \\epsilon, but needed when forming DNFs
    """
    L = set()
    for labels in language:
        if labels == "":
            continue
        for label in labels:
            L.add("!" + str(label))
    return "&".join(L)


def sample_reward_alphabet(X):
    """
    Returns the set of all reward values that appear in X
    """
    alphabet = set()
    alphabet.add(0.0)  # always includes 0
    for (_labels, rewards) in X:
        alphabet.update(rewards)
    return alphabet


def add_pvar(storage, storage_rev, used_pvars, subscript):
    """
    Records a propositional variable indexed with the subscript by assigning it a unique
    index used by the solver. Returns this index
    If the variable indexed with that subscript was already recorded, no mutation is done,
    while the index is still returned.
    """
    key = subscript
    pvar = storage_rev.get(key)
    if pvar is not None:
        return pvar
    used_pvars[0] += 1
    storage[used_pvars[0]] = subscript
    storage_rev[key] = used_pvars[0]
    return used_pvars[0]


def all_states_here(asdf, infer_termination):
    if infer_termination:
        return all_states_terminal(asdf)
    else:
        return all_states(asdf)


def all_states_terminal(n_states):
    return itertools.chain(all_states(n_states), [TERMINAL_STATE])


def all_states(n_states):
    return range(INITIAL_STATE, n_states)


# TODO build trie
def prefixes(X, without_terminal=False):
    yield ((), ())  # (\epsilon, \epsilon) \in Pref(X)
    for (labels, rewards) in X:
        ending = 1 if not without_terminal else 0
        for i in range(1, len(labels) + ending):
            yield (labels[0:i], rewards[0:i])


def convert_to_cxset(X, reward):
    r = set()
    for labels in X:
        labels = [['', c] for c in labels]
        labels = [c for cs in labels for c in cs]
        rewards = [0] * len(labels)
        rewards[-1] = reward
        r.add((tuple(labels), tuple(rewards)))
    return r


def sat_hyp(X_p, X_n, X_tl, n_states, infer_termination, report=True):
    X_p = convert_to_cxset(X_p, 1)
    X_n = convert_to_cxset(X_n, 0)
    X = X_p.union(X_n)
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    prop_d = dict()  # maps SAT's propvar (int) to (p: state, l: labels, q: state)
    prop_d_rev = dict()
    prop_o = dict()  # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
    prop_o_rev = dict()
    prop_x = dict()  # maps SAT's propvar (int) to (l: labels, q: state)
    prop_x_rev = dict()
    used_pvars = [0]  # p. var. counter
    g = Glucose4()  # solver

    # convenience methods
    def add_pvar_d(d):
        nonlocal prop_d
        nonlocal prop_d_rev
        return add_pvar(prop_d, prop_d_rev, used_pvars, d)

    def add_pvar_o(o):
        nonlocal prop_o
        nonlocal prop_o_rev
        return add_pvar(prop_o, prop_o_rev, used_pvars, o)

    def add_pvar_x(x):
        nonlocal prop_x
        nonlocal prop_x_rev
        return add_pvar(prop_x, prop_x_rev, used_pvars, x)

    # Encoding reward machines
    # (1)
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            g.add_clause([add_pvar_d((p, l, q))
                         for q in all_states_here(n_states, infer_termination)])
            for q1 in all_states_here(n_states, infer_termination):
                for q2 in all_states_here(n_states, infer_termination):
                    if q1 == q2:
                        continue
                    p_l_q1 = add_pvar_d((p, l, q1))
                    p_l_q2 = add_pvar_d((p, l, q2))
                    g.add_clause([-p_l_q1, -p_l_q2])

    # (2)
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            g.add_clause([add_pvar_o((p, l, r)) for r in reward_alphabet])
            for r1 in reward_alphabet:
                for r2 in reward_alphabet:
                    if r1 == r2:
                        continue
                    p_l_r1 = add_pvar_o((p, l, r1))
                    p_l_r2 = add_pvar_o((p, l, r2))
                    g.add_clause([-p_l_r1, -p_l_r2])

    # Consistency with sample
    # (3)
    # starts in the initial state
    g.add_clause([add_pvar_x((tuple(), INITIAL_STATE))])
    for p in all_states_here(n_states, infer_termination):
        if p == INITIAL_STATE:
            continue
        g.add_clause([-add_pvar_x((tuple(), p))])

    # (4)
    for (labels, _rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]  # type: ignore
        for p in all_states_here(n_states, infer_termination):
            for q in all_states_here(n_states, infer_termination):
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, q))
                x_2 = add_pvar_x((labels, q))
                g.add_clause([-x_1, -d, x_2])

    # (5)
    for (labels, rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]  # type: ignore
        r = rewards[-1]  # type: ignore
        for p in all_states_here(n_states, infer_termination):
            x = add_pvar_x((lm, p))
            o = add_pvar_o((p, l, r))
            g.add_clause([-x, o])

    # (Termination)
    if infer_termination:
        for (labels, _rewards) in prefixes(X, without_terminal=True):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]  # type: ignore
            x_2 = add_pvar_x((labels, TERMINAL_STATE))  # TODO REMOVE unneeded
            for p in all_states_here(n_states, infer_termination):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, TERMINAL_STATE))
                g.add_clause([-x_1, -d])

        for (labels, rewards) in X:
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            x_2 = add_pvar_x((labels, TERMINAL_STATE))  # TODO REMOVE unneeded
            for p in all_states_here(n_states, infer_termination):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, TERMINAL_STATE))
                d_t = -d if (labels, rewards) in X_tl else d
                g.add_clause([-x_1, d_t])

        for p in all_states_here(n_states, infer_termination):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = add_pvar_d((TERMINAL_STATE, l, p))
                g.add_clause([-d])

        for p in all_states_here(n_states, infer_termination):
            for l in language:
                o = add_pvar_o((TERMINAL_STATE, l, 0.0))
                g.add_clause([o])

    # found = False
    # # (Relevant events)
    # for relevant in powerset(language):
    #     assumptions = []
    #     for p in all_states_here(n_states):
    #         if p == TERMINAL_STATE:
    #             continue
    #         for l in language:
    #             if l in relevant:
    #                 continue
    #             d = add_pvar_d((p, l, p))
    #             o = add_pvar_o((p, l, 0.0))
    #             assumptions.extend([d, o])
    #     g.solve(assumptions=assumptions)
    #     # if len(relevant) == len(language):
    #     #     IPython.embed()
    #     if g.get_model() is None:
    #         continue
    #     else:
    #         found = True
    #         if report:
    #             print(f"found with assumptions {relevant}")
    #         break

    # if not found:
    #     return None

    g.solve()
    if g.get_model() is None:
        return None

    if report:
        print("found")

    # defaultdict(lambda: [None, None]) # maps (state, true_props) to (state, reward)
    transitions = dict()

    assert g != None

    for pvar in g.get_model():  # type: ignore
        if abs(pvar) in prop_d:
            if pvar > 0:
                (p, l, q) = prop_d[abs(pvar)]
                # assert transitions[(p, tuple(l))][0] is None
                if (p, tuple(l)) not in transitions:
                    transitions[(p, tuple(l))] = [None, None]
                transitions[(p, tuple(l))][0] = q
                # assert q is not None
        elif abs(pvar) in prop_o:
            if pvar > 0:
                (p, l, r) = prop_o[abs(pvar)]
                if (p, tuple(l)) not in transitions:
                    transitions[(p, tuple(l))] = [None, None]
                # assert transitions[(p, tuple(l))][1] is None
                transitions[(p, tuple(l))][1] = r
        elif abs(pvar) in prop_x:
            pass
        else:
            raise ValueError("Uknown p-var dict")

    g.delete()
    return transitions
