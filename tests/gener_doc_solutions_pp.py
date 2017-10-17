from automlk.solutions_pp import *
import pandas as pd

with open('../docs/solutions_pp.rst', 'w') as f:

    def output(x):
        f.write(x+'\n')
        print(x)

    def print_solutions(pp_type, text):
        output('%s:' % text)
        output('-'*len('%s:' % text))
        output('')
        l = []
        for s in pp_solutions:
            if s.pp_type == pp_type:
                output('**' + s.name + '**\n    *' + ", ".join([k for k in s.space_params.keys()])+'*')
                output('')

    print_solutions('categorical', 'categorical encoding')
    output('')
    print_solutions('text', 'text encoding')
    output('')
    print_solutions('missing', 'imputing missing values')
    output('')
    print_solutions('scaling', 'feature scaling')
    output('')
    print_solutions('feature', 'feature selection')
    output('')
