
# BestModel

This folder contains the best-performing model for the NLP coursework.

Contents:

- checkpoint_best/ : trained RoBERTa model weights
- dev.txt : predictions for official dev set
- test.txt : predictions for official test set
- best_threshold.txt : decision threshold
- best_temperature.txt : temperature scaling parameter
- run_config.txt : training hyperparameters
- notebook.ipynb : training notebook

Predictions follow the required format: one label per line (0 = No PCL, 1 = PCL).
