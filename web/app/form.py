from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SelectField, IntegerField, FileField, SelectMultipleField
from wtforms.validators import DataRequired
from automlk.metrics import metric_list
from automlk.dataset import get_dataset_list


class CreateDatasetForm(FlaskForm):
    # this is the form to create a dataset
    name = StringField(validators=[DataRequired()])
    domain = StringField(validators=[DataRequired()])
    description = TextAreaField()
    source = StringField()
    url = StringField()

    mode = SelectField(choices=[('standard', 'standard'), ('benchmark', 'benchmark'), ('competition', 'competition')],
                       default='standard')
    mode_file = SelectField(choices=[('upload', 'upload'), ('path', 'file path')], default='upload')

    filename_cols = StringField()
    file_cols = FileField()

    filename_train = StringField()
    file_train = FileField()

    filename_test = StringField()
    file_test = FileField()

    filename_submit = StringField()
    file_submit = FileField()


class UpdateDatasetForm(FlaskForm):
    # this is the form to update specific fields of a dataset
    name = StringField(validators=[DataRequired()])
    domain = StringField(validators=[DataRequired()])
    description = TextAreaField()
    is_uploaded = BooleanField()
    source = StringField()
    url = StringField()


class StartDatasetForm(FlaskForm):
    # this is the form to start the search on a dataset
    problem_type = SelectField(choices=[('classification', 'classification'), ('regression', 'regression')])
    regression_metric = SelectField(choices=[])
    regression_other_metrics = SelectMultipleField(choices=[(m.name, m.name) for m in metric_list if m.problem_type == 'regression'])
    classification_metric = SelectField(choices=[])
    classification_other_metrics = SelectMultipleField(choices=[(m.name, m.name) for m in metric_list if m.problem_type == 'classification'])

    y_col = SelectField(choices=[])
    col_submit = SelectField(choices=[])

    cv_folds = IntegerField(default=5)
    holdout_ratio = IntegerField(default=20)

    val_col = SelectField(choices=[])
    val_col_shuffle = BooleanField(default=True)

    sampling = BooleanField(default=False)

    def set_metrics_choices(self, specific_name, specific_content):
        if specific_content != '':
            # in this case, the only metrics available for primary metrics is specific
            self.regression_metric.choices = [('specific', specific_name + ' [specific]')]
            self.classification_metric.choices = self.regression_metric.choices
        else:
            self.regression_metric.choices = [(m.name, m.name) for m in metric_list if m.problem_type == 'regression']
            self.classification_metric.choices = [(m.name, m.name) for m in metric_list if m.problem_type == 'classification']

    def set_columns_choices(self, cols):
        self.y_col.choices = [(x, x) for x in cols]
        self.col_submit.choices = [(x, x) for x in cols]
        self.val_col.choices = [(x, x) for x in ['index'] + cols]


class ResetDatasetForm(FlaskForm):
    # form to confirm delete of a dataset
    reset_id = StringField('reset_id')
    reset_name = StringField('reset_name')
    reset_domain = StringField('reset_domain')
    reset_description = TextAreaField('reset_description')


class DeleteDatasetForm(FlaskForm):
    # form to confirm delete of a dataset
    id = StringField('id')
    name = StringField('name')
    domain = StringField('domain')
    description = TextAreaField('description')


class EditFeatureForm(FlaskForm):
    # form to edit a feature column of a dataset
    id = StringField('id')
    name = StringField('name')
    to_keep = SelectField(choices=[('False', 'No'), ('True', 'Yes')])
    domain = StringField('domain')
    description = TextAreaField('description')
    col_type = SelectField(choices=[('numerical', 'numerical'), ('categorical', 'categorical'), ('text', 'text'), ('date', 'date')])
    text_ref = SelectField(choices=[])

    def set_ref_choices(self, choices):
        self.text_ref.choices = [x for x in [('', '')] + list(choices)]


class EditFeatureEngineeringForm(FlaskForm):
    # form to edit function feature engineering
    content = TextAreaField('content')


class EditMetricsForm(FlaskForm):
    # form to edit function for specific metrics
    name = StringField('name', validators=[DataRequired()])
    best_is_min = BooleanField('best_is_min', default=True)
    content = TextAreaField('content')


class ConfigForm(FlaskForm):
    # form to configure set-up
    data = StringField('data')
    theme = SelectField(choices=[('bootswatch/3.3.7/darkly', 'darkly'),
                                 ('bootstrap/3.3.7/css', 'bootstrap'),
                                 ('bootswatch/3.3.7/flatly', 'flatly'),
                                 ('bootswatch/3.3.7/cerulean', 'cerulean'),
                                 ('bootswatch/3.3.7/cyborg', 'cyborg'),
                                 ('bootswatch/3.3.7/slate', 'slate'),
                                 ('bootswatch/3.3.7/solar', 'solar')
                                 ])
    bootstrap = StringField('bootstrap')
    graph_theme = SelectField(choices=[('dark', 'dark'),
                                       ('white', 'white'),
                                       ])
    store = SelectField(choices=[('redis', 'redis'), ('file', 'file')])
    store_url = StringField('store_url')


class ImportForm(FlaskForm):
    # form to import datasets
    file_import = FileField()


class DomainForm(FlaskForm):
    # form to select domain
    domain = SelectField(choices=[])

    def set_choices(self, choices):
        self.domain.choices = [(x, x) for x in [''] + list(choices)]


class DataForm(FlaskForm):
    # form to select domain
    col = SelectField(choices=[])

    def set_choices(self, choices):
        self.col.choices = [(x, x) for x in choices]


class CreateTextsetForm(FlaskForm):
    # this is the form to create a textset
    name = StringField(validators=[DataRequired()])
    description = TextAreaField()
    source = StringField()
    url = StringField()
    mode_file = SelectField(choices=[('upload', 'upload'), ('path', 'file path')], default='upload')
    filename = StringField()
    file_text = FileField()


class UpdateTextsetForm(FlaskForm):
    # this is the form to update specific fields of a textset
    name = StringField(validators=[DataRequired()])
    description = TextAreaField()
    source = StringField()
    url = StringField()


class ResetTextsetForm(FlaskForm):
    # form to confirm reset of a textset
    reset_id = StringField('id')
    reset_name = StringField('name')
    reset_description = TextAreaField('description')


class DeleteTextsetForm(FlaskForm):
    # form to confirm delete of a textset
    id = StringField('id')
    name = StringField('name')
    description = TextAreaField('description')


class DupplicateRound(FlaskForm):
    # for to dupplicate a dataset
    dataset = SelectField(choices=[])

    def set_choices(self, problem_type):
        self.dataset.choices = [(d.dataset_id, '#%s: %s' % (d.dataset_id, d.name)) for d in get_dataset_list() if d.problem_type == problem_type]