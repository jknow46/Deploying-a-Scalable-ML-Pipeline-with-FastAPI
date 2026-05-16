# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised machine learning classifier trained to predict whether an individual earns more than $50,000 per year based on U.S. Census demographic data.
It uses RandomForestClassifier (or whichever model you selected as your final optimized model) implemented with scikit‑learn.
The model is trained using a preprocessing pipeline that encodes categorical features and scales/normalizes numerical features where appropriate.
Artifacts saved include:

model.pkl — the trained classifier

encoder.pkl — the fitted OneHotEncoder used during preprocessing

The model is served through a FastAPI application for inference.

## Intended Use
The model is intended for educational purposes within the Udacity Machine Learning DevOps Engineer Nanodegree.
Its purpose is to demonstrate:

ML model training and evaluation

Data preprocessing and pipeline design

Deployment of a prediction API using FastAPI

CI/CD integration with GitHub Actions

Monitoring performance on data slices

It is not intended for real‑world financial, employment, or eligibility decisions.

## Training Data
The model is trained on the Census Income dataset (also known as the Adult dataset).
The dataset includes features such as:

Age

Workclass

Education

Marital status

Occupation

Race

Sex

Hours per week

Native country

The target variable is whether the individual earns > $50K annually.

The dataset was split into:

80% training data

20% test data

Categorical features were one‑hot encoded using the saved encoder.

## Evaluation Data
Evaluation was performed on the held‑out test set (20% of the original dataset).
Additionally, performance was evaluated on data slices, such as:

Race

Sex

Education

Workclass

Slice performance results are saved in slice_output.txt
## Metrics
The model was evaluated using:

Precision

Recall

F‑beta score (β = 1)

Accuracy

Your actual results (replace with your numbers if needed):

Accuracy: ~0.83

F1 Score: ~0.74

Slice metrics vary depending on the subgroup and are documented in slice_output.txt.

## Ethical Considerations
The Census dataset contains demographic attributes that are sensitive, including race, sex, and marital status.
Using such features in predictive models can:

Introduce or amplify bias

Lead to unfair or discriminatory outcomes

Misrepresent certain demographic groups

This model should not be used for real hiring, lending, or eligibility decisions.
Any real‑world deployment would require:

Bias audits

Fairness constraints

Continuous monitoring

Domain expert review
## Caveats and Recommendations
The model is trained on a dataset from the 1990s and may not generalize to modern populations.

The dataset contains known biases and imbalances.

Performance varies significantly across demographic slices.

The model should only be used for learning, experimentation, and demonstration.

For production use, consider:

More robust feature engineering

Fairness‑aware modeling

Updated datasets

Regular retraining and monitoring