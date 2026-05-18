Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

Model Details
This model is a supervised machine learning classifier trained to predict whether an individual earns more than $50,000 per year based on U.S. Census demographic data.
It uses a RandomForestClassifier implemented with scikit‑learn.
The model is trained using a preprocessing pipeline that applies one‑hot encoding to categorical features and label binarization to the target variable.

Artifacts saved include:

model.pkl — the trained classifier

encoder.joblib — the fitted OneHotEncoder used during preprocessing

lb.joblib — the fitted LabelBinarizer used for the target variable

The model is served through a FastAPI application for inference.

Intended Use
The model is intended for educational purposes within the Udacity Machine Learning DevOps Engineer Nanodegree.
Its purpose is to demonstrate:

ML model training and evaluation

Data preprocessing and pipeline design

Deployment of a prediction API using FastAPI

CI/CD integration with GitHub Actions

Monitoring performance on data slices

This model is not intended for real‑world financial, employment, or eligibility decisions.

Training Data
The model is trained on the Census Income (Adult) dataset.
The dataset includes features such as:

Age

Workclass

Education

Marital status

Occupation

Relationship

Race

Sex

Hours per week

Native country

The target variable is whether the individual earns > $50K annually.

The dataset was split into:

80% training data

20% test data

Categorical features were one‑hot encoded using the saved encoder.

Evaluation Data
Evaluation was performed on the held‑out 20% test set.
Additionally, performance was evaluated on data slices, including:

Race

Sex

Education

Workclass

Slice performance results are saved in slice_output.txt.

Metrics
The model was evaluated using precision, recall, and F1 score on the held‑out test set.

Precision: 0.7353

Recall: 0.6378

F1 Score: 0.6831

Slice‑level metrics vary across demographic groups and are documented in slice_output.txt.

Ethical Considerations
The Census dataset contains sensitive demographic attributes such as race, sex, and marital status.
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

Caveats and Recommendations
The dataset originates from the 1990s and may not generalize to modern populations.

The dataset contains known biases and imbalances.

Performance varies significantly across demographic slices.

The model should only be used for learning, experimentation, and demonstration.

For production use, consider:

More robust feature engineering

Fairness‑aware modeling

Updated datasets

Regular retraining and monitoring