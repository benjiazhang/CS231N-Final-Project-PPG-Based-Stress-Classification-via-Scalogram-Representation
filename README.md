# CS231N-Final-Project-PPG-Based-Stress-Classification-via-Scalogram-Representation
Wrist PPG-Based Stress Classification via Scalogram Representations and Transfer Learning




## Cross-Validation Outputs

The EfficientNet cross-validation script produces three output files:

### `crossval_results.csv`

Contains the raw results for every fold of every hyperparameter combination.

Each row corresponds to a single fold from a single hyperparameter configuration. Metrics are recorded from the epoch that achieved the highest validation F1 score within that fold.

Example:

| lr   | batch_size | dropout | fold | f1   | acc  | balanced_acc |
| ---- | ---------- | ------- | ---- | ---- | ---- | ------------ |
| 3e-4 | 16         | 0.3     | 0    | 0.81 | 0.84 | 0.82         |

This file is useful for:

* Inspecting fold-to-fold variability
* Computing standard deviations across folds
* Debugging unstable hyperparameter configurations

---

### `cv_summary.csv`

Contains the mean performance across all cross-validation folds for each hyperparameter combination.

The script groups rows from `crossval_results.csv` by hyperparameter settings and computes the average of:

* Loss
* Accuracy
* F1 score
* Balanced Accuracy

Example:

| lr   | batch_size | dropout | f1   |
| ---- | ---------- | ------- | ---- |
| 3e-4 | 16         | 0.3     | 0.81 |
| 1e-4 | 16         | 0.3     | 0.79 |

This file is used to compare hyperparameter configurations and select the best-performing model.

---

### `best_params.json`

Contains the single best hyperparameter configuration identified during cross-validation.

The script sorts `cv_summary.csv` by mean F1 score and saves the top-performing configuration.

Example:

```json
{
  "lr": 0.0003,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "dropout": 0.3,
  "epochs": 10
}
```

This file is intended to be used for final model training on the full training set.

---

### Model Selection Metric

Hyperparameter selection is based on the **mean validation F1 score** across all cross-validation folds.

Because the task is binary stress classification and class imbalance may be present, F1 score is used as the primary model selection metric. Balanced Accuracy is reported as a secondary metric to assess performance across both classes.

---

### Notes

* Cross-validation uses `GroupKFold` with subject IDs as groups.
* All windows from a given subject remain within the same fold.
* Validation metrics are recorded from the epoch with the highest validation F1 score within each fold.
* The final hyperparameter ranking is based on the mean fold performance reported in `cv_summary.csv`.
