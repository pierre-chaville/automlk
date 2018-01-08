from app import app
import os
from flask import send_file, jsonify
from automlk.dataset import get_dataset, get_dataset_folder
from automlk.results import create_predict_file
from automlk.context import get_config


def __path_data(dataset_id):
    folder = get_dataset_folder(dataset_id)
    if folder.startswith('..'):
        return os.path.abspath(folder)
    else:
        return folder


@app.route('/column/<string:dataset_id>/<string:col>', methods=['GET'])
def column(dataset_id, col):
    # gets the features
    dataset = get_dataset(dataset_id)
    if col == "None":
        return jsonify([f for f in dataset.features if f.to_keep and f.name != dataset.y_col][0].__dict__)
    for f in dataset.features:
        if f.name == col:
            return jsonify(f.__dict__)
    return jsonify()


@app.route('/get_img_dataset/<string:dataset_id>/<string:graph_type>', methods=['GET'])
def get_img_dataset(dataset_id, graph_type):
    # retrieves the graph at dataset level from dataset_id;round_id, where dataset_id is dataset id and round_id is round id
    if get_config()['graph_theme'] == 'dark':
        return send_file(__path_data(dataset_id) + '/graphs_dark/_%s.png' % graph_type, mimetype='image/png')
    else:
        return send_file(__path_data(dataset_id) + '/graphs/_%s.png' % graph_type, mimetype='image/png')


@app.route('/get_img_data/<string:dataset_id>/<string:col>', methods=['GET'])
def get_img_data(dataset_id, col):
    # retrieves the graph of col data
    if get_config()['graph_theme'] == 'dark':
        return send_file(__path_data(dataset_id) + '/graphs_dark/_col_%s.png' % col, mimetype='image/png')
    else:
        return send_file(__path_data(dataset_id) + '/graphs/_col_%s.png' % col, mimetype='image/png')


@app.route('/get_img_round/<string:dataset_id>/<string:round_id>/<string:graph_type>', methods=['GET'])
def get_img_round(dataset_id, round_id, graph_type):
    # retrieves the graph at dataset level from dataset_id;round_id, where dataset_id is dataset id and round_id is round id
    if get_config()['graph_theme'] == 'dark':
        return send_file(__path_data(dataset_id) + '/graphs_dark/%s_%s.png' % (graph_type, round_id),
                         mimetype='image/png')
    else:
        return send_file(__path_data(dataset_id) + '/graphs/%s_%s.png' % (graph_type, round_id), mimetype='image/png')


@app.route('/get_doc_html/<string:dataset_id>', methods=['GET'])
def get_doc(dataset_id):
    # return html doc as zip file
    return send_file(__path_data(dataset_id) + '/doc.zip', as_attachment=True)


@app.route('/get_doc_pdf/<string:dataset_id>', methods=['GET'])
def get_doc_pdf(dataset_id):
    # retrieves the pdf document of the dataset
    return send_file(__path_data(dataset_id) + '/docs/_build/latex/dataset.pdf', as_attachment=True)


@app.route('/get_predict/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_predict(dataset_id, round_id):
    # download the prediction file
    create_predict_file(dataset_id, round_id)
    return send_file(__path_data(dataset_id) + '/submit/predict_%s.xlsx' % round_id,
                     as_attachment=True, attachment_filename='predict_%s_%s.xlsx' % (dataset_id, round_id))


@app.route('/get_pipeline/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_pipeline(dataset_id, round_id):
    # download the pipeline file (pickle)
    return send_file(__path_data(dataset_id) + '/models/%s_pipe_model.pkl' % round_id,
                     as_attachment=True, attachment_filename='pipe_model_%s_%s.pkl' % (dataset_id, round_id))


@app.route('/get_model/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_model(dataset_id, round_id):
    # download the model file (pickle)
    return send_file(__path_data(dataset_id) + '/models/%s_model.pkl' % round_id,
                     as_attachment=True, attachment_filename='model_%s_%s.pkl' % (dataset_id, round_id))


@app.route('/get_submit/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_submit(dataset_id, round_id):
    # download the submit file
    return send_file(__path_data(dataset_id) + '/submit/submit_%s.csv' % round_id,
                     as_attachment=True, attachment_filename='submit_%s_%s.csv' % (dataset_id, round_id))


@app.route('/get_explain/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_explain(dataset_id, round_id):
    # download the explanation html
    return send_file(__path_data(dataset_id) + '/predict/eli5_model_%s.html' % round_id, as_attachment=False)
