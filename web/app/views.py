from app import app
import time
import os
import uuid
from flask import render_template, send_file, redirect, request, abort, flash, send_from_directory, jsonify
from automlk.results import *
from automlk.doc import gener_doc
from .form import *
from automlk.context import get_uploads_folder, get_dataset_folder, get_config, set_config
from automlk.dataset import get_dataset_list, get_dataset, delete_dataset, update_dataset, reset_dataset, \
    update_feature_dataset, get_dataset_sample
from automlk.worker import get_search_rounds
from automlk.graphs import get_cnf_matrix
from automlk.store import set_key_store
from automlk.dataset import create_dataset
from automlk.monitor import get_heart_beeps


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # home page: list of models
    # datasets = get_dataset_list(include_results=True)[::-1]
    datasets = get_home_best()
    sel_form = DomainForm()
    domains = set([d.domain for d in datasets])
    sel_form.set_choices(domains)
    del_form = DeleteDatasetForm()
    reset_form = ResetDatasetForm()
    if request.method == 'POST':
        datasets = [d for d in datasets if d.domain.startswith(sel_form.domain.data)]
    return render_template('index.html', datasets=datasets, domains=domains, refresher=int(time.time()),
                           sel_form=sel_form, reset_form=reset_form, del_form=del_form, config=get_config())


@app.route('/start/<string:dataset_id>', methods=['GET'])
def start(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'searching')
    return redirect('/index')


@app.route('/pause/<string:dataset_id>', methods=['GET'])
def pause(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'pause')
    return redirect('/index')


@app.route('/gendoc/<string:dataset_id>', methods=['GET'])
def gendoc(dataset_id):
    dataset = get_dataset(dataset_id)
    gener_doc(dataset)
    return redirect('/dataset/%s#doc' % dataset_id)


@app.route('/dataset/<string:dataset_id>', methods=['GET'])
def dataset(dataset_id):
    # zoom on a specific dataset
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    doc_path = os.path.abspath(get_dataset_folder(dataset_id) + '/docs/_build/html/index.html')
    doc_pdf = os.path.abspath(get_dataset_folder(dataset_id) + '/docs/_build/latex/dataset.pdf')
    form = EditFeatureForm()
    data_form = DataForm()
    data_form.set_choices([f.name for f in dataset.features if f.to_keep and f.name != dataset.y_col])
    sample = get_dataset_sample(dataset_id)
    if not os.path.exists(doc_path):
        doc_path = ''
    if not os.path.exists(doc_pdf):
        doc_pdf = ''
    if len(search) > 0:
        best = get_best_models(dataset_id)
        best_pp = get_best_pp(dataset_id)
        # separate models (level 0) from ensembles (level 1)
        best1 = [b for b in best if b['level'] == 1]
        best2 = [b for b in best if b['level'] == 2]
        return render_template('dataset.html', dataset=dataset, best1=best1, best2=best2, best_pp=best_pp,
                               n_searches1=len(search[search.level == 1]), n_searches2=len(search[search.level == 2]),
                               form=form, data_form=data_form, doc_path=doc_path, doc_pdf=doc_pdf,
                               sample=sample, refresher=int(time.time()), config=get_config())
    else:
        return render_template('dataset.html', dataset=dataset, n_searches1=0, doc_path=doc_path, form=form,
                               data_form=data_form, sample=sample, refresher=int(time.time()), config=get_config())


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


@app.route('/details/<string:prm>', methods=['GET'])
def details(prm):
    # list of searches for a specific type of model
    dataset_id, model = prm.split(';')
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    cols, best = get_best_details(search, model)
    best = best.to_dict(orient='records')[:10]
    return render_template('details.html', dataset=dataset, model=model, best=best, cols=cols,
                           refresher=int(time.time()), config=get_config())


@app.route('/details_pp/<string:prm>', methods=['GET'])
def details_pp(prm):
    # list of searches for a specific type of pre-processing
    dataset_id, process = prm.split(';')
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    cols, best = get_best_details_pp(search, process)
    best = best.to_dict(orient='records')[:10]
    return render_template('details_pp.html', dataset=dataset, process=process, best=best, cols=cols,
                           refresher=int(time.time()), config=get_config())


@app.route('/round/<string:prm>', methods=['GET'])
def round(prm):
    # details of a search round (1 pre-processing + 1 model configuration)
    dataset_id, round_id = prm.split(';')
    dataset = get_dataset(dataset_id)
    search = get_search_rounds(dataset.dataset_id)
    round = search[search.round_id == int(round_id)].to_dict(orient='records')[0]
    pipeline = round['pipeline']
    if len(pipeline) < 1:
        pipeline = []
    else:
        # exclude passthrough and no scaling for display
        pipeline = [s for s in pipeline if s[0] not in ['NO-SCALE', 'PASS']]
    params = get_round_params(search, round_id)
    features = get_feature_importance(dataset.dataset_id, round_id)
    y_names, cnf_matrix, sums_matrix = get_cnf_matrix(dataset_id, round_id, 'eval')
    return render_template('round.html', dataset=dataset, round=round, pipeline=pipeline,
                           features=features, params=params, cols=params.keys(), refresher=int(time.time()),
                           y_names=y_names, cnf_matrix=cnf_matrix, sums_matrix=sums_matrix, config=get_config())


def __path_data(dataset_id):
    folder = get_dataset_folder(dataset_id)
    if folder.startswith('..'):
        return os.path.abspath(folder)
    else:
        return folder


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
        return send_file(__path_data(dataset_id) + '/graphs_dark/%s_%s.png' % (graph_type, round_id), mimetype='image/png')
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


@app.route('/get_submit/<string:dataset_id>/<string:round_id>', methods=['GET'])
def get_submit(dataset_id, round_id):
    # download the submit file
    return send_file(__path_data(dataset_id) + '/submit/submit_%s.csv' % round_id,
                     as_attachment=True, attachment_filename='submit_%s_%s.csv' % (dataset_id, round_id))


@app.route('/create', methods=['GET', 'POST'])
def create():
    # form to create a new dataset
    form = CreateDatasetForm()
    if request.method == 'POST':
        r = create_dataset_form(form)
        if r:
            return r
    return render_template('create.html', form=form, config=get_config())


def check_upload_file(f):
    # check and upload a file
    if f.filename == '':
        return ''
    if f.filename == '':
        flash('file %s type must be csv, xls or xlsx' % f.filename)
        return ''
    ext = f.filename.split('.')[-1].lower()
    if ext not in ['csv', 'tsv', 'xls', 'xlsx']:
        flash('file %s type must be csv, tsv, xls or xlsx' % f.filename)
        return ''
    else:
        upload = get_uploads_folder() + '/' + str(uuid.uuid4()) + '.' + ext
        f.save(upload)
        return upload


def create_dataset_form(form):
    # performs creation of a dataset from a form
    if form.validate():
        #try:
        if form.mode_file.data == 'upload':
            form.filename_cols.data = check_upload_file(form.file_cols.data)
            form.filename_train.data = check_upload_file(form.file_train.data)
            if form.mode.data == 'benchmark':
                form.filename_test.data = check_upload_file(form.file_test.data)
            else:
                form.filename_test.data = ''
            if form.mode.data == 'competition':
                form.filename_submit.data = check_upload_file(form.file_submit.data)
            else:
                form.filename_submit.data = ''

        create_dataset(name=form.name.data,
                       domain=form.domain.data,
                       description=form.description.data,
                       source=form.source.data,
                       url=form.url.data,
                       problem_type=form.problem_type.data,
                       metric=form.metric.data,
                       other_metrics=form.other_metrics.data,
                       mode=form.mode.data,
                       filename_train=form.filename_train.data,
                       holdout_ratio=form.holdout_ratio.data / 100.,
                       filename_cols=form.filename_cols.data,
                       filename_test=form.filename_test.data,
                       filename_submit=form.filename_submit.data,
                       col_submit=form.col_submit.data,
                       cv_folds=form.cv_folds.data,
                       y_col=form.y_col.data,
                       val_col=form.val_col.data,
                       val_col_shuffle=form.val_col_shuffle.data,
                       sampling=form.sampling.data)
        return redirect('index')
        #except Exception as e:
        #    flash(e)
    else:
        flash(", ".join([key + ': ' + form.errors[key][0] for key in form.errors.keys()]))
    return None


@app.route('/duplicate/<dataset_id>', methods=['GET', 'POST'])
def duplicate(dataset_id):
    # form to duplicate a dataset
    form = CreateDatasetForm()

    if request.method == 'POST':
        r = create_dataset_form(form)
        if r:
            return r
    else:
        dataset = get_dataset(dataset_id)

        # copy data to form
        form.name.data = dataset.name + ' (copy)'
        form.domain.data = dataset.domain
        form.description.data = dataset.description
        form.source.data = dataset.source
        form.url.data = dataset.url
        form.problem_type.data = dataset.problem_type
        form.metric.data = dataset.metric
        form.other_metrics.data = ",".join(dataset.other_metrics)
        form.mode.data = dataset.mode
        form.mode_file.data = 'path'
        form.filename_train.data = dataset.filename_train
        form.holdout_ratio.data = int(dataset.holdout_ratio * 100)
        form.filename_cols.data = dataset.filename_cols
        form.filename_test.data = dataset.filename_test
        form.filename_submit.data = dataset.filename_submit
        form.col_submit.data = dataset.col_submit
        form.cv_folds.data = dataset.cv_folds
        form.y_col.data = dataset.y_col
        form.val_col.data = dataset.val_col
        form.val_col_shuffle.data = dataset.val_col_shuffle
        form.sampling.data = dataset.sampling

    return render_template('create.html', form=form, config=get_config())


@app.route('/update/<dataset_id>', methods=['GET', 'POST'])
def update(dataset_id):
    # form to update a dataset
    form = UpdateDatasetForm()
    if request.method == 'POST':
        if form.validate():
            update_dataset(dataset_id,
                           name=form.name.data,
                           domain=form.domain.data,
                           description=form.description.data,
                           is_uploaded=form.is_uploaded.data,
                           source=form.source.data,
                           url=form.url.data)
            return redirect('/index')
    else:
        dataset = get_dataset(dataset_id)

        # copy data to form
        form.name.data = dataset.name
        form.domain.data = dataset.domain
        form.description.data = dataset.description
        form.source.data = dataset.source
        form.url.data = dataset.url
    return render_template('update.html', form=form, config=get_config())


@app.route('/edit_feature/<string:dataset_id>', methods=['POST'])
def edit_feature(dataset_id):
    # edit the description of a column
    form = EditFeatureForm()
    if form.validate():
        update_feature_dataset(dataset_id,
                               name=form.name.data,
                               description=form.description.data,
                               to_keep=form.to_keep.data,
                               col_type=form.col_type.data)
        return redirect('/dataset/%s#features' % dataset_id)


@app.route('/reset', methods=['POST'])
def reset():
    # reset a dataset
    form = ResetDatasetForm()
    if form.validate():
        reset_dataset(form.reset_id.data)
    return redirect('/index')


@app.route('/delete', methods=['POST'])
def delete():
    # delete a dataset
    form = DeleteDatasetForm()
    if form.validate():
        delete_dataset(form.id.data)
    return redirect('/index')


@app.route('/import', methods=['GET', 'POST'])
def import_file():
    # form to import a file of dataset descriptions
    form = ImportForm()
    if request.method == 'POST':
        if form.validate():
            f = form.file_import.data
            ext = f.filename.split('.')[-1].lower()
            if ext not in ['csv', 'tsv', 'xls', 'xlsx']:
                flash('file type must be csv, tsv, xls or xlsx')
            else:
                if ext == 'csv':
                    df = pd.read_csv(f)
                elif ext == 'tsv':
                    df = pd.read_csv(f, sep='\t', header=0)
                else:
                    df = pd.read_excel(f)
                try:
                    line_number = 1
                    for line in df.fillna('').to_dict(orient='records'):
                        print('creating dataset %s in %s' % (line['name'], line['problem_type']))
                        create_dataset(**line)
                        line_number += 1
                    return redirect('index')
                except Exception as e:
                    flash('Error in line %d: %s' % (line_number, str(e)))
        else:
            print('not validated')

    return render_template('import.html', form=form, config=get_config())


@app.route('/monitor', methods=['GET'])
def monitor():
    # monitor workers
    return render_template('monitor.html', controller=get_heart_beeps('controller'),
                           workers=get_heart_beeps('worker'), config=get_config())


@app.route('/config', methods=['GET', 'POST'])
def config():
    # view/edit configuration
    form = ConfigForm()
    if request.method == 'POST':
        if form.validate():
            try:
                set_config(data=form.data.data,
                           theme=form.theme.data,
                           bootstrap=form.bootstrap.data,
                           graph_theme=form.graph_theme.data,
                           store=form.store.data,
                           store_url=form.store_url.data)
            except Exception as e:
                flash(str(e))
    else:
        config = get_config()

        # copy data to form
        form.data.data = config['data']
        form.theme.data = config['theme']
        form.bootstrap.data = config['bootstrap']
        form.graph_theme.data = config['graph_theme']
        form.store.data = config['store']
        form.store_url.data = config['store_url']

    return render_template('config.html', form=form, config=get_config())
