from app import app
import time
from flask import render_template, send_file, redirect, request, abort
from .helper import *
from automlk.context import get_dataset_folder
from automlk.dataset import get_dataset_list, get_dataset
from automlk.search import get_search_rounds
from automlk.graphs import graph_history_search
from automlk.store import set_key_store

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # home page: list of models
    datasets = get_dataset_list(include_status=True)[::-1]
    return render_template('index.html', datasets=datasets, refresher=int(time.time()))


@app.route('/start/<string:dataset_id>', methods=['GET'])
def start(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'searching')
    return redirect('/index')


@app.route('/pause/<string:dataset_id>', methods=['GET'])
def pause(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'pause')
    return redirect('/index')


@app.route('/dataset/<string:dataset_id>', methods=['GET', 'POST'])
def dataset(dataset_id):
    # zoom on a specific dataset
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    if len(search) > 0:
        best = get_best_models(search)
        # separate models (level 0) from ensembles (level 1)
        best1 = best[best.level == 1]
        best2 = best[best.level == 2]
        graph_history_search(dataset, search, best1, 1)
        graph_history_search(dataset, search, best2, 2)
        return render_template('dataset.html', dataset=dataset, best1=best1.to_dict(orient='records'),
                               best2=best2.to_dict(orient='records'),
                               n_searches1=len(search[search.level == 1]),
                               n_searches2=len(search[search.level == 2]),
                               refresher=int(time.time()))
    else:
        return render_template('dataset.html', dataset=dataset, n_searches1=0, refresher=int(time.time()))

# TODO: graph per parameter


@app.route('/details/<string:prm>', methods=['GET', 'POST'])
def details(prm):
    # list of searches for a specific type of model
    dataset_id, model = prm.split(';')
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    cols, best = get_best_details(search, model)
    best = best.to_dict(orient='records')[:10]
    return render_template('details.html', dataset=dataset, model=model, best=best, cols=cols,
                           refresher=int(time.time()))


@app.route('/round/<string:prm>', methods=['GET', 'POST'])
def round(prm):
    # details of a search round (1 preprocessing + 1 model configuration)
    dataset_id, round_id = prm.split(';')
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    round = search[search.round_id == int(round_id)].to_dict(orient='records')[0]
    steps = get_process_steps(round['process_steps'])
    params = get_round_params(search, round_id)
    features = get_feature_importance(dataset.dataset_id, round_id)
    return render_template('round.html', dataset=dataset, round=round, steps=steps, features=features, params=params,
                           cols=params.keys(), refresher=int(time.time()))


@app.route('/get_img_dataset/<string:prm>', methods=['GET'])
def get_img_dataset(prm):
    # retrieves the graph at dataset level from dataset_id;round_id, where dataset_id is dataset id and round_id is round id
    dataset_id, graph_type = prm.split(';')
    return send_file(get_dataset_folder(dataset_id) + '/graphs/_%s.png' % graph_type, mimetype='image/png')


@app.route('/get_img_round/<string:prm>', methods=['GET'])
def get_img_round(prm):
    # retrieves the graph at dataset level from dataset_id;round_id, where dataset_id is dataset id and round_id is round id
    dataset_id, round_id, graph_type = prm.split(';')
    return send_file(get_dataset_folder(dataset_id) + '/graphs/%s_%s.png' % (graph_type, round_id), mimetype='image/png')
