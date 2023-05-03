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
    rewrites = ast.rewrites()
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
    # src = '((.)* > door&secret > (base)+ > base&documents&secret) & (!detected)*'
    # src = '((.)* > door&secret > (enemy&base)+ > enemy&base&documents&secret > (.)* > friendly&base) & (!detected)*'
    # src = '(!rock)* > ((goldmine)* > rock&goldmine)+'
    # src = '((.)* > coffee > (.)* > mail | (.)* > mail > (.)* > coffee) > (.)* > office'

    # {}, {'base_1'}, {}, {'base_2'}, {}, {'base_3'}, {}
    # {}, {'base_2'}, {}, {'base_3'}, {}, {'base_1'}, {}
    # {}, {'base_2'}, {}, {'base_3'}, {}, {'base_1'}, {}, {'base_2'}, {}, {'base_3'}
    # src = '((!base_2&!base_3)* > base_1 > (!base_1&!base_3)* > base_2 > (!base_1&!base_2)* > base_3)+ | ((!base_1&!base_3)* > base_2 > (!base_1&!base_2)* > base_3 > (!base_2&!base_3)* > base_1)+ | ((!base_1&!base_2)* > base_3 > (!base_2&!base_3)* > base_1 > (!base_1&!base_3)* > base_2)+'
    # src = '((!base_2&!base_3)+ > base_1 > (!base_1&!base_3)+ > base_2 > (!base_1&!base_2)+ > base_3)+ | ((!base_1&!base_3)+ > base_2 > (!base_1&!base_2)+ > base_3 > (!base_2&!base_3)+ > base_1)+ | ((!base_1&!base_2)+ > base_3 > (!base_2&!base_3)+ > base_1 > (!base_1&!base_3)+ > base_2)+'
    # rm = compiler_interface.compile(src)

    # src = '(.)* > (goldmine)* > rock&goldmine > ((goldmine)* > rock&goldmine)*'

    # HERE
    # component by component compilation, figure out where the error is
    # the error may very unlikely be in the rm class. unlikely because it already shows on the NFA when visualized

    # src = 'A'
    # src = '(A)*'
    # src = 'B'
    # src = '(A)* > B'
    # src = '((A)* > B)*'
    # src = '((A)* > B)+'
    # src = 'A > (A)*'
    # src = '(A)*'
    # src = '(A & D & B) | (A & D & C)'
    # src = 'A & D & (B | C)'
    # src = 'A > (B | C)'  # -> A > B | A > C
    # src = '(B | C) > A'  # -> B > A | C > A
    # src = 'A > (B | C) > D'
    # src = 'B > A | C > A | D'  # -> (B | C) > A | D

    # src = '(A)~'
    # src = '((A)~)*'
    # src = '((.)* > ($A | $B | $C))+'

    src = '(.)* > ((A)* > B&A){##N}'
    # src = 'A > (B){3} > C'

    # In [7]: rm({}, {'equipment'}, {}, {'mail'}, {}, {'wall'}, {}, {'door'})
    # Out[7]: [0, 0, 0, 0, 0, 0, 0, 1]

    # In [6]: rm({}, {'equipment'}, {}, {}, {}, {'mail'}, {}, {'door'})
    # Out[6]: [0, 0, 0, 0, 0, 0, 0, 0]

    # In [5]: rm({}, {'equipment'}, {}, {'wall'}, {}, {'mail'}, {}, {'door'})
    # Out[5]: [0, 0, 0, 0, 0, 0, 0, 1]

    ast = compiler_interface.parse(src)
    nfa, _ = compiler_interface.get_nfa(src)
    # visualize_compilestate(nfa, src)
    dfa, _ = compiler_interface.get_dfa(src)
    rm = compiler_interface.compile(src)
    IPython.embed()  # type: ignore


def main():
    test_compiler()


if __name__ == "__main__":
    main()
