import IPython

import compiler_interface
from tools import produce_datasets
from visualization import visualize_compilestate, visualize_rm, visualize_ast


def test_compiler():
    # src = 'A'
    # src = '!_'
    # src = 'A > B'
    # src = 'A | B'
    # src = '!A | !B'
    # src = '!(!A | !B)'
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
    src = '((.)* > door&secret > (enemy&base)+ > enemy&base&documents&secret > (.)* > friendly&base) & (!detected)*'

    # tokens = compiler_interface.lex(src)
    # print(tokens)

    # ast = compiler_interface.parse(src)
    # visualize_ast(ast, f'AST: {src}')

    # nfa, node_creator = compiler_interface.get_nfa(src)
    # visualize_compilestate(nfa, f'NFA: {src}')

    # dfa, node_creator = compiler_interface.get_dfa(src)
    # visualize_compilestate(dfa, f'DFA: {src}')

    rm = compiler_interface.compile(src)
    # visualize_rm(rm, f'RM: {src}')

    IPython.embed()  # type: ignore


def main():
    test_compiler()


if __name__ == "__main__":
    main()
