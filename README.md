# General Info


This is a public repository, serving as a demo. The organization of the repo is only to show example data science project structure and workflow. Contributions for both the code quality and the data science approaches are well appreciated. 


# Tree Structure


```
│├── pros_project
│    ├── src
│        ├── data
│            ├── imports
│                    ├── *.csv
│                    └── *.parquet
│            ├── exports
│                └── features
│                    └── *.pkl
│                ├── figures
│                    └── *.svg
│                ├── models
│                    └── *.pkl
│        ├── notebooks
│            └── *.ipynb
│        └── utils
│            └── base_classes.py
│    ├── .gitignore
│    ├── README.md
│    └── README_env.md
```


# Data Source


Publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/rhythmghai/250k-customer-churn-prediction-dataset). It represents examplary features to mimic clients' banking activity for binary, churn prediction. Note that it is a synthetic dataset and in reality the churn modelling tasks are more challenging than the obtained results in this project.