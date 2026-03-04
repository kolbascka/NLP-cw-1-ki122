# NLP Coursework 1 - PCL Detection

Author: Katia Ignashova
Imperial College London - Natural Language Processing
Leaderboard name: **katIgn**

This repository contains the code and artefacts for my submission to the NLP coursework on **Patronizing and Condescending Language (PCL) detection**. The task is based on the *Don't Patronize Me!* dataset introduced in the SemEval shared task.

The project implements a **RoBERTa-based multi-task learning approach** for detecting PCL in news paragraphs. The model jointly learns:

1. **Binary PCL detection** (primary task)
2. **7-category multi-label classification** of PCL phenomena (auxiliary task)

The auxiliary task is applied only to paragraphs labeled as PCL and helps the model learn richer representations of patronizing language.

---

# Repository Structure

```
NLP-cw-1-ki122/
│
├── BestModel/
│   ├── checkpoint_best/          # Final trained model checkpoint
│   ├── dev.txt                   # Predictions for official dev set
│   ├── test.txt                  # Predictions for official test set
│   ├── best_threshold.txt        # Decision threshold selected on internal validation
│   ├── best_temperature.txt      # Temperature used for calibration
│   ├── run_config.txt            # Hyperparameters of the selected run
│   └── nlp-cw1-final-notebook.ipynb   # Full training + evaluation notebook
│
└── README.md
```

The `BestModel/` directory contains all artefacts required for marking and reproduction of the final model.

---

# Model Overview

The proposed system extends the RoBERTa baseline using **joint multi-task learning**:

Shared encoder:

```
RoBERTa-base
```

Task heads:

```
Binary classifier → predicts PCL presence
Multi-label classifier → predicts 7 PCL categories
```

Training objective:

```
L = L_binary + λ L_category
```

where:

* `L_binary` = class-weighted cross-entropy (handles class imbalance)
* `L_category` = binary cross-entropy over 7 labels
* `λ` = auxiliary task weight

Additional techniques used:

* **Class-weighted loss** to address the ~9.5% positive class rate
* **Threshold optimisation** on an internal validation split
* **Temperature scaling** for probability calibration
* **Maximum sequence length = 160 tokens** (based on EDA)

The final selected configuration achieved:

```
Positive-class F1 (official dev set): 0.6173
```

---

# Running the Notebook

All experiments are implemented in:

```
BestModel/nlp-cw1-final-notebook.ipynb
```

The notebook reconstructs the official train/dev splits, trains the model, selects the best checkpoint, and generates `dev.txt` and `test.txt`.

Required Python packages:

```
transformers
datasets
scikit-learn
torch
numpy
pandas
matplotlib
```

---

# Required Data Structure

To reproduce the results, the repository expects the following directory layout:

```
./data/
│
├── dontpatronizeme_categories.tsv
├── dontpatronizeme_pcl.tsv
├── task4_test.tsv
│
└── practice_splits/
    ├── train_semeval_parids-labels.csv
    └── dev_semeval_parids-labels.csv
```

These files correspond to the official dataset and shared-task splits.

---

# Reproducing the Results

1. Place the dataset files in the `./data/` directory following the structure above.
2. Open the notebook:

```
BestModel/nlp-cw1-final-notebook.ipynb
```

3. Run all cells sequentially.

The notebook will:

* reconstruct the official splits
* train multiple model configurations
* select the best checkpoint
* generate predictions for the dev and test sets

The final predictions are written to:

```
dev.txt
test.txt
```

---

# Notes

* The repository includes the **final trained checkpoint** used for submission.
* The notebook outputs are preserved to show the exact results reported in the coursework report.
* All model selection steps are performed using an **internal validation split**, while the official dev set is used only for evaluation.
* In the notebook, there exist some cells that were applicable mainly to my development process (e.g., mounting Google Drive as I worked on Google Colab). You may need to amend these for perfect replication.

---

# Reference

Perez-Almendros et al. (2022)
*Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language Towards Vulnerable Communities.*

SemEval Shared Task on Patronizing Language Detection.
