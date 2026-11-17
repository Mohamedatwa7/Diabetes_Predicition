# Diabetes Prediction App

A machine-learning model that predicts diabetes risk from health indicators, served as an interactive Streamlit app.

## Features

The app (`diabetes_streamlit.py`) has six tabs:

- **Prediction:** enter patient indicators and get a prediction with class probabilities.
- **EDA:** distributions and relationships across the dataset.
- **Feature engineering:** how the inputs were prepared and scaled.
- **Dataset description:** what each feature means.
- **Model insights:** confusion matrix, ROC curve and AUC, and the decision tree structure.

The full analysis and training process is in `ML2DIABETESFINAL.ipynb`.

## Run it

```bash
pip install -r requirements.txt
streamlit run diabetes_streamlit.py
```

## Stack

Python · pandas · scikit-learn · Streamlit · matplotlib · seaborn
