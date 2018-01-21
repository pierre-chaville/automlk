from app import app
from flask import render_template, request, flash
from .form import *
from automlk.monitor import get_heart_beeps
from automlk.context import get_config, set_config


@app.route('/monitor', methods=['GET'])
def monitor():
    # monitor workers
    return render_template('monitor.html', controller=get_heart_beeps('controller'),
                           grapher=get_heart_beeps('grapher'), worker_text=get_heart_beeps('worker_text'),
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
