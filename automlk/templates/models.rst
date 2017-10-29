**Best results found in {{best|count}} models and {{n_searches}} configurations**

.. csv-table:: Best models
   :header: "model", "metrics", "other", "params"
   :stub-columns: 1
   :widths: 20, 20, 20, 30

    {% for r in best %}
    "{{ r.model_name}}", "cv max: {{ print_score(r.cv_max) }}, cv: {{print_score(r.cv_mean) }} +/- {{ print_score_std(r.cv_std) }}, test: {{ print_score(r.score_test) }}", "{{ print_other_metrics(r.eval_other_metrics) }}", "{{ print_params(r.model_params) }}" {% endfor %}
