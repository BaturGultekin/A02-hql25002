# A02-hql25002 - Ping Pong Regression
This repository contains my OPIM 5512 Assignment 02 submission.

The project uses the California Housing dataset from `sklearn` to train an `MLPRegressor` model and generate actual vs. predicted plots for the train and test sets.

My assigned partner dropped the class :/, so I completed the assignment individually while still using branches and pull requests to document the workflow (even includes my delusional looking comments and reviews xd). 

## Student Information
- Student: Batur Gultekin (hql25002)
- Partner: N/A, assigned partner dropped the class :(

## Project Workflow

The project was completed through small pull requests:

1. Load California Housing dataset and create train/test split
2. Add `MLPRegressor` with early stopping
3. Add train predictions and train actual vs. predicted plot
4. Add test predictions and test actual vs. predicted plot
5. Improve plot labels, formatting, and model hyperparameters

## Model
The final model uses:
- `MLPRegressor`
- `early_stopping=True`
- `hidden_layer_sizes=(16, 8)`
- `batch_size=128`
- `activation="relu"`
- `validation_fraction=0.2`

The best validation score improved from `0.695` to `0.744` after updating the hidden layer structure and batch size.

## Repository Structure
```text
.
├── README.md
├── src
│   └── ds_pipeline.py
└── figures
    ├── train_actual_vs_pred.png
    └── test_actual_vs_pred.png
```

## How to Run
Activate the course environment:
```bash
conda activate OPIM5512
```

Run the script:
```bash
python src/ds_pipeline.py
```

Or run it directly through conda:
```bash
conda run -n OPIM5512 python src/ds_pipeline.py
```
The script trains the model and saves the output plots in the `figures/` folder.

## Output Figures
The repository includes:
- `figures/train_actual_vs_pred.png`
- `figures/test_actual_vs_pred.png`
Both figures show actual vs. predicted `MedHouseVal` values with a reference line.