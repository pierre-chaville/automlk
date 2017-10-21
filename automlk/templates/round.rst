**{{ round.round_id }}**

Round info
__________


cv max
    {{ print_score(round.cv_max) }}

score eval
    {{ print_score(round.score_eval) }}

score test
    {{ print_score(round.score_test) }}

cv
    {{ print_score(round.cv_mean) }} +/- {{ print_score(round.cv_std) }}

eval metrics
    {{ print_other_metrics(round.eval_other_metrics) }}

test metrics
    {{ print_other_metrics(round.test_other_metrics) }}

model
    {{ round.solution_name }}

time
    {{ round.start_time }}

pre-processing duration
    {{ print_duration(round.duration_process ) }}

modeling duration
    {{ print_duration(round.duration_model) }}

rounds
    {{ round.num_rounds }}

host
    {{ round.host_name }}

round_id
    {{ round.round_id }}

model level
    {{ round.level }}

pre-processing
    {{ round.process_steps }}

params
    {{ round.model_params }}

.. csv-table:: Parameters
   :header: "param", "value"

   {% for col in cols %}
   "{{ col }}", "{{params[col]}}" {% endfor %}


Pre-processing steps
____________________

{% for step in pipeline %}
{{ step[1] }} {{ step[2] }}
    {{ print_params(step[3]) }}

{% endfor %}

Feature importance
__________________

{% if features|count > 0 %}

.. csv-table:: Parameters
   :header: "feature", "importance"

    {% for f in features[:100] %}{% if f["pct_importance"] > 0.005 %}
    "{{ f["feature"] }}", {{ f["pct_importance"] }}{% endif %}{% endfor %}
{% else %}
This model has no feature importance.
{% endif %}

Prediction
__________

.. figure:: ../graphs/predict_eval_{{round.round_id}}.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    prediction on eval set


.. figure:: ../graphs/predict_test_{{round.round_id}}.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    prediction on test set


Prediction
__________

.. figure:: ../graphs/hist_eval_{{round.round_id}}.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    prediction on eval set


.. figure:: ../graphs/hist_test_{{round.round_id}}.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    prediction on test set


