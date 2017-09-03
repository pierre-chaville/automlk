from app import app
import time
from flask import render_template, send_file, redirect, request, abort
from .helper import *
from automlk.context import get_dataset_folder
from automlk.dataset import get_dataset_list, get_dataset
from automlk.search import get_search_rounds
from automlk.graphs import graph_history_search


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # home page: list of models
    datasets = get_dataset_list()[::-1]
    return render_template('index.html', datasets=datasets, refresher=int(time.time()))


@app.route('/dataset/<string:uid>', methods=['GET', 'POST'])
def dataset(uid):
    # zoom on a specific dataset
    dataset = get_dataset(uid)
    search = get_search_rounds(dataset.uid)
    if len(search) > 0:
        best = get_best_models(search)
        # separate models (level 0) from ensembles (level 1)
        best0 = best[best.model_level == 0]
        best1 = best[best.model_level == 1]
        graph_history_search(dataset, search, best0, 0)
        graph_history_search(dataset, search, best1, 1)
        return render_template('dataset.html', dataset=dataset, best0=best0.to_dict(orient='records'),
                               best1=best1.to_dict(orient='records'),
                               n_searches0=len(search[search.model_level == 0]),
                               n_searches1=len(search[search.model_level == 1]),
                               uid=uid, refresher=int(time.time()))
    else:
        return render_template('dataset.html', dataset=dataset, n_searches0=0, uid=uid, refresher=int(time.time()))

# TODO: graph per parameter


@app.route('/details/<string:prm>', methods=['GET', 'POST'])
def details(prm):
    # list of searches for a specific type of model
    uid, model = prm.split(';')
    dataset = get_dataset(uid)
    search = get_search_rounds(dataset.uid)
    cols, best = get_best_details(search, model)
    best = best.to_dict(orient='records')[:10]
    return render_template('details.html', dataset=dataset, model=model, best=best, cols=cols,
                           refresher=int(time.time()))


@app.route('/round/<string:prm>', methods=['GET', 'POST'])
def round(prm):
    # details of a search round (1 preprocessing + 1 model configuration)
    uid, uuid = prm.split(';')
    dataset = get_dataset(uid)
    search = get_search_rounds(dataset.uid)
    round = search[search.uuid == uuid].to_dict(orient='records')[0]
    steps = get_process_steps(round['process'])
    params = get_round_params(search, uuid)
    features = get_feature_importance(dataset.uid, uuid)
    return render_template('round.html', dataset=dataset, round=round, steps=steps, features=features, params=params,
                           cols=params.keys(), refresher=int(time.time()))


@app.route('/get_img_dataset/<string:prm>', methods=['GET'])
def get_img_dataset(prm):
    # retrieves the graph at dataset level from uid;uuid, where uid is dataset id and uuid is round id
    uid, graph_type = prm.split(';')
    return send_file(get_dataset_folder(uid) + '/graphs/_%s.png' % graph_type, mimetype='image/png')


@app.route('/get_img_round/<string:prm>', methods=['GET'])
def get_img_round(prm):
    # retrieves the graph at dataset level from uid;uuid, where uid is dataset id and uuid is round id
    uid, uuid, graph_type = prm.split(';')
    return send_file(get_dataset_folder(uid) + '/graphs/%s_%s.png' % (graph_type, uuid), mimetype='image/png')
