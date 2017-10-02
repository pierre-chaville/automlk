from flask_wtf import Form
from wtforms import StringField, BooleanField, SelectField, IntegerField, DecimalField
from wtforms.validators import DataRequired
from automlk.metrics import metric_list


class DatasetForm(Form):
    # this is the form to create a dataset

    name = StringField(validators=[DataRequired()])
    description = StringField()
    problem_type = SelectField(choices=[('classification', 'classification'), ('regression', 'regression')])
    y_col = StringField(validators=[DataRequired()])

    is_uploaded = BooleanField()
    source = StringField()

    filename_cols = StringField()
    filename_train = StringField()
    filename_test = StringField()
    filename_submit = StringField()
    col_submit = StringField()

    metric = SelectField(choices=[(m.name, m.name) for m in metric_list])
    other_metrics = StringField()

    cv_folds = IntegerField(default=5)
    holdout_ratio = IntegerField(default=20)
    val_col = StringField()
    val_col_shuffle = BooleanField()

    is_public = BooleanField()
    url = StringField()
