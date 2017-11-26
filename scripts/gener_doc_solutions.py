from automlk.solutions import *
import pandas as pd

with open('../docs/solutions.rst', 'w') as f:

    def output(x):
        f.write(x+'\n')
        print(x)

    def print_solutions(level, problem_type):
        output('%s:' % problem_type)
        output('_'*len('%s:' % problem_type))
        l = []
        for s in model_solutions:
            if s.level == level and s.problem_type == problem_type:
                output('**' + s.name + '**\n    *' + ", ".join([k for k in s.space_params.keys()])+'*')
                output('')

    output('Models level 1')
    output('--------------')
    output('')
    print_solutions(1, 'regression')
    output('')
    print_solutions(1, 'classification')
    output('')
    output('Ensembles')
    output('---------')
    output('')
    print_solutions(2, 'regression')
    output('')
    print_solutions(2, 'classification')