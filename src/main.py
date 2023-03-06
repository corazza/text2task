import IPython

import compiler_interface
from regex_validation import equivalent
from tools import produce_datasets
from visualization import visualize_compilestate, visualize_rm, visualize_ast
from regex_printer import expr_to_str


def test_rewrites_equivalence(src: str, n_rewrites: int):
    ast = compiler_interface.parse(src)
    dfa, _ = compiler_interface.get_dfa(src)
    appears = ast.appears()
    rewrites = ast.rewrites(appears, n_rewrites)
    print(f'testing equivalence for: {src}')
    for rewrite in rewrites:
        src_b = expr_to_str(rewrite)
        dfa_b, _ = compiler_interface.get_dfa(src_b)
        ineq_evidence = equivalent(appears, dfa, dfa_b, 5, 10)
        if len(ineq_evidence) > 0:
            print()
            print(f'failed: {src_b}')
            for evidence in ineq_evidence:
                print(evidence)


def test_compiler():
    # src = 'A'
    # src = '!_'
    # src = 'A > B'
    # src = 'A | B'
    # src = '!A | !B'
    # src = '((A)~ | (B)~)~'
    # src = 'C > (A | B)'
    # src = 'C > (A & B)'
    # src = 'C > C&!(A | B) > D'
    # src = '((A | B) > C) & (D)+'
    # src = '(A&!B)*'

    # src = '(!A)*'

    # src = '((!B)*)&(A)*'
    # src = '((B)*)~&(A)*'
    # src = 'A > _ > B > (.)*'

    src = '((.)* > door&secret > (base)+ > base&documents&secret) & (!detected)*'
    # src = '((.)* > door&secret > (enemy&base)+ > enemy&base&documents&secret > (.)* > friendly&base) & (!detected)*'
    # src = '(!rock)* > ((goldmine)* > rock&goldmine)+'
    # src = '((.)* > coffee > (.)* > mail | (.)* > mail > (.)* > coffee) > (.)* > office'

    # tokens = compiler_interface.lex(src)
    # print(tokens)

    # nfa, node_creator = compiler_interface.get_nfa(src)
    # visualize_compilestate(nfa, f'NFA: {src}')

    # dfa, node_creator = compiler_interface.get_dfa(src)
    # visualize_compilestate(dfa, f'DFA: {src}')

    # rm = compiler_interface.compile(src)
    # visualize_rm(rm, f'RM: {src}')

    # HERE
    # In [4]: rm({}, {}, {'goldmine'}, {'goldmine'}, {'goldmine', 'rock'}, {'rock'}, {'goldmine', 'rock'}, {'goldmine'}, {'goldmine'})
    # Out[4]: [0, 0, 0, 0, 1, 0, 1, 0, 0]

    # IPython.embed()  # type: ignore

    test_rewrites_equivalence(src, 10)

    # src = 'A | B'

    # ast = compiler_interface.parse(src)
    # appears = ast.appears()
    # dfa, _ = compiler_interface.get_dfa(src)
    # dfa_b, _ = compiler_interface.get_dfa(src)

    # for evidence in equivalent(appears, dfa, dfa_b, 5, 10):
    #     print(evidence)

    # visualize_compilestate(dfa, 'a')
    # visualize_compilestate(dfa_b, 'b')


def main():
    test_compiler()


if __name__ == "__main__":
    main()
