import os
import pandas as pd
import numpy as np
from .worker import get_importance
from .context import get_dataset_folder
from .results import *
from automlk.worker import get_search_rounds
from automlk.graphs import graph_history_search
from .print import *
import jinja2
import subprocess


jinja_globals = {'print_list': print_list,
                 'print_score': print_score,
                 'print_score_std': print_score_std,
                 'print_value': print_value,
                 'print_duration': print_duration,
                 'print_params': print_params,
                 'print_other_metrics': print_other_metrics,
                 }


def render(template, fileout, **kwargs):
    """
    generates output from template into the fileout file
    :param template: jinja2 template to be used (in folder /template)
    :param fileout: file to store the results
    :param kwargs: args to render the template
    :return:
    """
    t = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="../automlk/templates/")).get_template(template)
    with open(fileout, 'w') as f:
        f.write(t.render({**kwargs, **jinja_globals}))


def gener_doc(dataset):
    """
    generate the documentation of this dataset

    :param dataset: dataset object
    :return:
    """
    # check or create doc folder
    folder = get_dataset_folder(dataset.dataset_id) + '/doc'
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder + '/_build')
        os.makedirs(folder + '/_static')
        os.makedirs(folder + '/_templates')

    # generate conf.py
    render('conf.txt', folder + '/conf.py', dataset=dataset)
    render('make.bat', folder + '/make.bat', dataset=dataset)
    render('makefile.txt', folder + '/Makefile', dataset=dataset)

    # generate index
    render('index.rst', folder + '/index.rst', dataset=dataset)

    # dataset data and features
    search = get_search_rounds(dataset.dataset_id)
    if len(search) > 0:
        best = get_best_models(search)
        best_pp = get_best_pp(search.copy())
        # separate models (level 0) from ensembles (level 1)
        best1 = best[best.level == 1]
        best2 = best[best.level == 2]
        graph_history_search(dataset, search, best1.copy(), 1)
        graph_history_search(dataset, search, best2.copy(), 2)

        render('dataset.rst', folder + '/dataset.rst', dataset=dataset, best1=best1.to_dict(orient='records'),
               best2=best2.to_dict(orient='records'), best_pp=best_pp,
               n_searches1=len(search[search.level == 1]),
               n_searches2=len(search[search.level == 2]))

        # then for the best rounds
        N_ROUNDS = 5
        for round_id in list(best1.round_id.values[:N_ROUNDS]) + list(best2.round_id.values[:N_ROUNDS]):
            round = search[search.round_id == int(round_id)].to_dict(orient='records')[0]
            pipeline = [s for s in round['pipeline'] if s[0] not in ['NO-SCALE', 'PASS']]
            params = get_round_params(search, round_id)
            features = get_feature_importance(dataset.dataset_id, round_id)
            render('round.rst', folder + '/round_%s.rst' % round_id, dataset=dataset, round=round,
                   pipeline=pipeline, features=features, params=params, cols=params.keys())
    else:
        # return render_template('dataset.html', dataset=dataset, n_searches1=0)
        render('dataset.rst', folder + '/dataset.rst', dataset=dataset, n_searches1=0)

    # then generate html and pdf with make
    # ls_output = subprocess.check_output(['ls'])
    # print(ls_output)
    # subprocess.call(['dir;cd ' + folder + ';dir;make html'], shell=True)
    #os.system('cd ' + folder + ';dir;make html')

