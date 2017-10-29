from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SelectField, IntegerField, FileField
from wtforms.validators import DataRequired
from automlk.metrics import metric_list


class CreateDatasetForm(FlaskForm):
    # this is the form to create a dataset
    name = StringField(validators=[DataRequired()])
    domain = StringField(validators=[DataRequired()])
    description = TextAreaField()
    problem_type = SelectField(choices=[('classification', 'classification'), ('regression', 'regression')])
    y_col = StringField(validators=[DataRequired()])

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
    col_submit = StringField()

    metric = SelectField(choices=[(m.name, m.name) for m in metric_list])
    other_metrics = StringField()

    cv_folds = IntegerField(default=5)
    holdout_ratio = IntegerField(default=20)
    val_col = StringField(default='index')
    val_col_shuffle = BooleanField(default=True)


class UpdateDatasetForm(FlaskForm):
    # this is the form to update specific fields of a dataset
    name = StringField(validators=[DataRequired()])
    domain = StringField(validators=[DataRequired()])
    description = TextAreaField()
    is_uploaded = BooleanField()
    source = StringField()
    url = StringField()


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
    graph_style = SelectField(choices=[('default', 'default'),
                                       ('bmh', 'bmh'),
                                       ('ggplot', 'ggplot'),
                                       ('seaborn', 'seaborn'),
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
        self.domain.choices = [(x, x) for x in set([''] + choices)]
