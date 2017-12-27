from app import app
import time
import os
import uuid
from flask import render_template, redirect, request, flash
from automlk.results import *
from automlk.doc import gener_doc
from .form import *
from automlk.context import get_uploads_folder
from automlk.config import DUPLICATE_QUEUE
from automlk.dataset import *
from automlk.specific import *
from automlk.worker import get_search_rounds
from automlk.graphs import get_cnf_matrix
from automlk.store import set_key_store

# include additional views
from .views_api import *
from .views_textset import *
from .views_specific import *
from .views_admin import *


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
        redirect_to_index = redirect('/index')
        response = app.make_response(redirect_to_index)
        response.set_cookie('automlk_folder', value=sel_form.domain.data)
        return response
    if 'automlk_folder' in request.cookies:
        folder = request.cookies.get('automlk_folder')
        domains = [d for d in domains if d.startswith(folder)]
        datasets = [d for d in datasets if d.domain in domains]
        sel_form.domain.data = folder
    return render_template('index.html', datasets=datasets, domains=domains, refresher=int(time.time()),
                           sel_form=sel_form, reset_form=reset_form, del_form=del_form, config=get_config())


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
    form.set_ref_choices([(t.textset_id, t.name) for t in get_textset_list()])
    data_form = DataForm()
    data_form.set_choices([f.name for f in dataset.features if f.to_keep and f.name != dataset.y_col])
    fe_content = get_feature_engineering(dataset_id)
    metrics_name, metrics_best, metrics_content = get_specific_metrics(dataset_id)
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
                               form=form, data_form=data_form,
                               doc_path=doc_path, doc_pdf=doc_pdf, fe_content=fe_content,
                               metrics_name=metrics_name, metrics_best=metrics_best, metrics_content=metrics_content,
                               sample=sample, refresher=int(time.time()), config=get_config())
    else:
        return render_template('dataset.html', dataset=dataset, n_searches1=0, doc_path=doc_path, form=form,
                               data_form=data_form, fe_content=fe_content,
                               metrics_name=metrics_name, metrics_best=metrics_best, metrics_content=metrics_content,
                               sample=sample, refresher=int(time.time()), config=get_config())


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


@app.route('/round/<string:prm>', methods=['GET', 'POST'])
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
    form = DupplicateRound()
    form.set_choices(dataset.problem_type)
    if request.method == 'POST':
        # apply round parameters for searching in another dataset
        lpush_key_store(DUPLICATE_QUEUE, {'round': round, 'dataset': form.dataset.data})
    return render_template('round.html', dataset=dataset, round=round, pipeline=pipeline, form=form,
                           features=features, params=params, cols=params.keys(), refresher=int(time.time()),
                           y_names=y_names, cnf_matrix=cnf_matrix, sums_matrix=sums_matrix, config=get_config())


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
        try:
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
                           mode=form.mode.data,
                           filename_train=form.filename_train.data,
                           filename_cols=form.filename_cols.data,
                           filename_test=form.filename_test.data,
                           filename_submit=form.filename_submit.data)
            return redirect('index')
        except Exception as e:
            flash(e)
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
        form.mode.data = dataset.mode
        form.mode_file.data = 'path'
        form.filename_train.data = dataset.filename_train
        form.filename_cols.data = dataset.filename_cols
        form.filename_test.data = dataset.filename_test
        form.filename_submit.data = dataset.filename_submit

    return render_template('create.html', form=form, config=get_config())


@app.route('/start_search/<dataset_id>', methods=['GET', 'POST'])
def start_search(dataset_id):
    # form to duplicate a dataset
    dataset = get_dataset(dataset_id)
    specific_name, specific_best, specific_content = get_specific_metrics(dataset_id)
    form = StartDatasetForm()
    form.set_metrics_choices(specific_name, specific_content)
    form.set_columns_choices([f.name for f in dataset.features])
    if request.method == 'POST':
        if form.validate():
            if form.problem_type.data == 'regression':
                metric = form.regression_metric.data
                other_metrics = form.regression_other_metrics.data
            else:
                metric = form.classification_metric.data
                other_metrics = form.classification_other_metrics.data
            if dataset.mode == 'competition':
                col_submit = form.col_submit.data
            else:
                col_submit = ''
            update_problem_dataset(dataset_id,
                                   problem_type=form.problem_type.data,
                                   metric=metric,
                                   other_metrics=other_metrics,
                                   holdout_ratio=form.holdout_ratio.data/100,
                                   col_submit=col_submit,
                                   cv_folds=form.cv_folds.data,
                                   y_col=form.y_col.data,
                                   val_col=form.val_col.data,
                                   val_col_shuffle=form.val_col_shuffle.data,
                                   sampling=form.sampling.data)
            set_key_store('dataset:%s:status' % dataset_id, 'searching')
            return redirect('index')
        else:
            flash(", ".join([key + ': ' + form.errors[key][0] for key in form.errors.keys()]))
    else:
        # copy data to form
        form.problem_type.data = dataset.problem_type
        if dataset.problem_type == 'regression':
            form.regression_metric.data = dataset.metric
            form.regression_other_metrics.data = dataset.other_metrics
        else:
            form.classification_metric.data = dataset.metric
            form.classification_other_metrics.data = dataset.other_metrics

        form.holdout_ratio.data = int(dataset.holdout_ratio * 100)
        form.col_submit.data = dataset.col_submit
        form.cv_folds.data = dataset.cv_folds
        form.y_col.data = dataset.y_col
        form.val_col.data = dataset.val_col
        form.val_col_shuffle.data = dataset.val_col_shuffle
        form.sampling.data = dataset.sampling

    return render_template('start.html', dataset=dataset, form=form, config=get_config())


@app.route('/restart/<string:dataset_id>', methods=['GET'])
def restart(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'searching')
    return redirect('/index')


@app.route('/pause/<string:dataset_id>', methods=['GET'])
def pause(dataset_id):
    set_key_store('dataset:%s:status' % dataset_id, 'pause')
    return redirect('/index')


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
        print('post')
        if form.validate():
            print('validate')
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
                    return redirect('/')
                except Exception as e:
                    flash('Error in line %d: %s' % (line_number, str(e)))
        else:
            print('not validated')

    return render_template('import.html', form=form, config=get_config())

