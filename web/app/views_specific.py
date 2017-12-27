from app import app
from flask import render_template, redirect, request, flash
from .form import *
from automlk.dataset import *
from automlk.specific import *


@app.route('/edit_feature/<string:dataset_id>', methods=['POST'])
def edit_feature(dataset_id):
    # edit the description of a column
    form = EditFeatureForm()
    form.set_ref_choices([(t.textset_id, t.name) for t in get_textset_list()])
    if form.validate():
        update_feature_dataset(dataset_id,
                               name=form.name.data,
                               description=form.description.data,
                               to_keep=form.to_keep.data,
                               col_type=form.col_type.data,
                               text_ref=form.text_ref.data,
                               )
        return redirect('/dataset/%s#features' % dataset_id)


@app.route('/edit_engineering/<string:dataset_id>', methods=['GET', 'POST'])
def edit_engineering(dataset_id):
    # edit the content of feature engineering of a dataset
    form = EditFeatureEngineeringForm()
    if request.method == 'POST':
        if form.validate():
            # try to execute the function
            try:
                exec_feature_engineering(form.content.data)
                update_feature_engineering(dataset_id, content=form.content.data)
                return redirect('/dataset/%s#specific' % dataset_id)
            except Exception as e:
                flash(str(e))
    else:
        content = get_feature_engineering(dataset_id)
        form.content.data = content

    return render_template('edit_engineering.html', form=form, dataset_id=dataset_id, config=get_config())


@app.route('/delete_engineering/<string:dataset_id>', methods=['GET', 'POST'])
def delete_engineering(dataset_id):
    # delete the content of feature engineering of a dataset
    delete_feature_engineering(dataset_id)
    return redirect('/dataset/%s#specific' % dataset_id)


@app.route('/edit_metrics/<string:dataset_id>', methods=['GET', 'POST'])
def edit_metrics(dataset_id):
    # edit the content of specific metrics for a dataset
    form = EditMetricsForm()
    if request.method == 'POST':
        if form.validate():
            # try to execute the function
            try:
                exec_specific_metrics(form.content.data)
                update_specific_metrics(dataset_id,
                                        name=form.name.data,
                                        best_is_min=form.best_is_min.data,
                                        content=form.content.data)
                return redirect('/dataset/%s#specific' % dataset_id)
            except Exception as e:
                flash(str(e))
    else:
        name, best_is_min, content = get_specific_metrics(dataset_id)
        form.content.data = content
        form.name.data = name

    return render_template('edit_metrics.html', form=form, dataset_id=dataset_id, config=get_config())


@app.route('/delete_metrics/<string:dataset_id>', methods=['GET', 'POST'])
def delete_metrics(dataset_id):
    # delete specific metrics of a dataset
    delete_specific_metrics(dataset_id)
    return redirect('/dataset/%s#specific' % dataset_id)

