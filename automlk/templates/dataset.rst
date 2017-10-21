The Dataset
===========

*id: {{dataset.dataset_id}}*

{{ dataset.name }}
__________________

{{dataset.description}}

The dataset {{ dataset.name }} ({{dataset.url}}) is used in {{dataset.mode}} mode.
It is a **{{ dataset.problem_type }}** problem on column *{{dataset.y_col}}* {% if dataset.problem_type == 'classification' %}with {{dataset.y_n_classes}} classes ({{ print_list(dataset.y_class_names[:10]) }}){% endif %}.

The dataset has {{dataset.n_rows*1000}} rows, {{dataset.n_cols}} columns, out of which {{dataset.n_cat_cols}} categorical columns and {{dataset.text_cols|count}} text columns.
There are {{dataset.n_missing}} columns with {{dataset.n_missing}} missing values.

The metric to optimize for this dataset is **{{dataset.metric}}**{% if dataset.other_metrics|count > 0 %} and the other metrics are {{ print_list(dataset.other_metrics) }}{% endif %}.

The cross validation consists of **{{dataset.cv_folds}} folds**, the split is on the {{dataset.val_col}} column {% if dataset.val_col_shuffle %}shuffled{% else %}**not shuffled**{% endif %}.
{% if dataset.with_test_set %}
The test set is used for test.
{% else %}
The training set has been splited with a holdout out ratio of {{dataset.holdout_ratio * 100}}% into eval set ({{100 - dataset.holdout_ratio * 100}}%) and test set ({{dataset.holdout_ratio * 100}}%).
{% endif %}

Train set: {{dataset.filename_train}}

{% if dataset.filename_test != '' %}
Test set: {{dataset.filename_test}}
{% endif %}
{% if dataset.filename_submit != '' %}
Submit set: {{dataset.filename_submit}}
Column submit: {{dataset.col_submit}}
{% endif %}


Features
--------

X columns:
    {{ print_list(dataset.x_cols) }}


{% if dataset.cat_cols|count > 0 %}
Categorical columns
    {{ print_list(dataset.cat_cols) }}
{% endif %}

{% if dataset.text_cols|count > 0 %}
Text columns
    {{ print_list(dataset.text_cols) }}
{% endif %}

.. csv-table:: List of features
   :header: "name", "description", "keep", "raw type", "type", "missing", "unique", "values"
   :widths: 10, 20, 10, 10, 10, 10, 10, 20

    {% for col in dataset.features %}
    "{{col.name}}", "{{col.description}}", "{{col.to_keep}}", "{{col.raw_type}}", "{{col.col_type}}", {{col.n_missing}}, {{col.n_unique_values}}, "{{col.first_unique_values[:200]}}" {% endfor %}


Data distribution
-----------------

.. figure:: ../graphs/_hist_train_{{dataset.y_col}}.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    distribution of the training set


Feature correlation
-------------------

.. figure:: ../graphs/_correl.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    correlation of the features in the training set



{% if n_searches1 > 0 %}
Results of the search of best models
====================================

Search history with models level 1
----------------------------------

.. figure:: ../graphs/_history_1.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    graph of the best result over time


.. figure:: ../graphs/_models_1.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    graph of the best result over time from the 5 best models

{% if best2|count > 0 %}

Search history with ensemble models
-----------------------------------

.. figure:: ../graphs/_history_2.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    graph of the best result over time


.. figure:: ../graphs/_models_2.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    graph of the best result over time from the 5 best models
{% endif %}

Best models level 1
-------------------

{% set best = best1 %}
{% set n_searches = n_searches1 %}
{% include 'models.rst' %}


Best pre-processing steps
-------------------------

.. csv-table:: Best pre-processing
   :header: "process", "cv max", "eval", "test", "cv", "other", "#", "duration", "params"
   :widths: 10, 10, 10, 10, 10, 10, 10, 10, 20

    {% for cat, best_cat_pp in best_pp %}
    **{{ cat }}**{% for r in best_cat_pp %}
    "{{ r.cat_name}}", {{ print_score(r.cv_max) }}, {{ print_score(r.score_eval) }}, {{ print_score(r.score_test) }}, "{{print_score(r.cv_mean) }} +/- {{ print_score_std(r.cv_std) }}", "{{ print_other_metrics(r.eval_other_metrics) }}", {{ r.searches}}, "{{print_duration(r.duration_process) }}, {{print_duration(r.duration_model) }}", "{{ print_params(r.cat_params) }}" {% endfor %}{% endfor %}


{% if best2|count > 0 %}

Best ensemble models
--------------------

{% set best = best2 %}
{% set n_searches = n_searches2 %}
{% include 'models.rst' %}
{% endif %}

{% for round in best1[:5] %}
.. include:: round_{{round.round_id}}.rst
{% endfor %}

{% endif %}


