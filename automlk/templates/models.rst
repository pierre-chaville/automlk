**Best results found in {{best|count}} models and {{n_searches}} configurations**

.. csv-table:: Best models
   :header: "model", "cv max", "eval", "test", "cv", "other", "#", "duration", "params"
   :widths: 10, 10, 10, 10, 10, 10, 10, 10, 20

    {% for r in best %}
    "{{ r.model_name}}", {{ print_score(r.cv_max) }}, {{ print_score(r.score_eval) }}, {{ print_score(r.score_test) }}, "{{print_score(r.cv_mean) }} +/- {{ print_score_std(r.cv_std) }}", "{{ print_other_metrics(r.eval_other_metrics) }}", {{ r.searches}}, "{{print_duration(r.duration_process) }}, {{print_duration(r.duration_model) }}", "{{ print_params(r.model_params) }}" {% endfor %}
