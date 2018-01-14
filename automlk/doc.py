import os
import sys
import glob
import zipfile
import pandas as pd
import numpy as np
from .context import get_dataset_folder
from .results import *
from automlk.worker import get_search_rounds
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
                 'print_title': print_title,
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
    folder = get_dataset_folder(dataset.dataset_id) + '/docs'
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
        best = get_best_models(dataset.dataset_id)
        best_pp = get_best_pp(dataset.dataset_id)
        # separate models (level 0) from ensembles (level 1)
        best1 = [b for b in best if b['level'] == 1]
        best2 = [b for b in best if b['level'] == 2]
        print(len(best1), len(best2))
        print(best1[:2])
        render('dataset.rst', folder + '/dataset.rst', dataset=dataset, best1=best1, best2=best2, best_pp=best_pp,
               n_searches1=len(search[search.level == 1]),
               n_searches2=len(search[search.level == 2]))

        # then for the best rounds
        N_ROUNDS = 5
        for round_id in list([b['round_id'] for b in best1[:N_ROUNDS]]) + list([b['round_id'] for b in best2[:N_ROUNDS]]):
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
    if sys.platform == 'linux':
        subprocess.call(['sh', '../scripts/gen_doc.sh', os.path.abspath(get_dataset_folder(dataset.dataset_id)+'/docs')])
    else:
        os.system('call ../scripts/gen_doc ' + os.path.abspath(get_dataset_folder(dataset.dataset_id)+'/docs'))

    # generate zip file of the html site
    with zipfile.ZipFile(get_dataset_folder(dataset.dataset_id) + '/doc.zip', 'w') as z:
        root = get_dataset_folder(dataset.dataset_id) + '/docs/_build/html/'
        for dir in ['', '_static/', '_images/', '_sources/']:
            for f in glob.glob(root + dir + '*.*'):
                z.write(f, dataset.dataset_id + '/' + dir + os.path.basename(f))
