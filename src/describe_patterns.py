patterns_then_2 = ['first do A, then do B',
                   'after doing A, do B',
                   'before doing B, do A',
                   'do A, but not before B']
patterns_then_3 = ['first do A, then do B, finally do C',
                   'A then B then C']
patterns_then_4 = ['first do A, then do B, then do C, then do D',
                   'A then B then C then D']
patterns_then = [patterns_then_2, patterns_then_3, patterns_then_4]

patterns_or_2 = ['first do A, then do B',
                 'after doing A, do B',
                 'before doing B, do A',
                 'do A, but not before B']
patterns_or_3 = ['first do A, then do B, finally do C',
                 'A then B then C']
patterns_or_4 = ['first do A, then do B, then do C, then do D',
                 'A then B then C then D']
patterns_or = [patterns_or_2, patterns_or_3, patterns_or_4]

patterns_repeat_1 = ['repeat A']
patterns_repeat = [patterns_repeat_1]

# (COFFEE MAIL | MAIL COFFEE)* OFFICE

patterns = {'(A B | B A)': ['A and B'],
            '(A)*': ['repeat: A']}
