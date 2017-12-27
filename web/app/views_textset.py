from app import app
from flask import render_template, send_file, redirect, request, abort, flash, send_from_directory, jsonify
from .form import *
from automlk.textset import *
from automlk.context import get_uploads_folder


@app.route('/textset_list', methods=['GET', 'POST'])
def text_list():
    # list of text sets
    textset_list = get_textset_list()
    del_form = DeleteTextsetForm()
    return render_template('text_list.html', textset_list=textset_list, del_form=del_form,
                           refresher=int(time.time()), config=get_config())


@app.route('/textset/<string:textset_id>', methods=['GET', 'POST'])
def text_set(textset_id):
    # detail of a text set
    textset = get_textset(textset_id)
    return render_template('text.html', textset=textset, refresher=int(time.time()), config=get_config())


@app.route('/create_text', methods=['GET', 'POST'])
def create_text():
    # form to create a new textset
    form = CreateTextsetForm()
    if request.method == 'POST':
        if form.validate():
            # try:
            if form.mode_file.data == 'upload':
                # check and upload a file
                filename = form.file_text.data.filename
                if filename == '' or filename.split('.')[-1].lower() != 'txt':
                    flash('file %s type must be txt' % filename)
                else:
                    form.filename.data = get_uploads_folder() + '/' + str(uuid.uuid4()) + '.' + filename.split('.')[
                        -1].lower()
                    form.file_text.data.save(form.filename.data)

            create_textset(name=form.name.data,
                           description=form.description.data,
                           source=form.source.data,
                           url=form.url.data,
                           filename=form.filename.data)
            return redirect('/textset_list')
            # except Exception as e:
            #    flash(e)
        else:
            flash(", ".join([key + ': ' + form.errors[key][0] for key in form.errors.keys()]))

    return render_template('create_text.html', form=form, config=get_config())


@app.route('/update_text/<textset_id>', methods=['GET', 'POST'])
def update_text(textset_id):
    # form to update a textset
    form = UpdateTextsetForm()
    if request.method == 'POST':
        if form.validate():
            update_textset(textset_id,
                           name=form.name.data,
                           description=form.description.data,
                           source=form.source.data,
                           url=form.url.data)
            return redirect('/textset_list')
    else:
        textset = get_textset(textset_id)
        # copy data to form
        form.name.data = textset.name
        form.description.data = textset.description
        form.source.data = textset.source
        form.url.data = textset.url
    return render_template('update_text.html', form=form, config=get_config())


@app.route('/delete_text', methods=['POST'])
def delete_text():
    # delete a textset
    form = DeleteTextsetForm()
    if form.validate():
        delete_textset(form.id.data)
    return redirect('/textset_list')

