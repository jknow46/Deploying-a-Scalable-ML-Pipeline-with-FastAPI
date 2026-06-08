import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    # Example simple feature engineering:
    # - One-hot encode categorical
    # - TF-IDF on 'name'
    X = X.copy()

    if "name" in X.columns:
        tfidf = TfidfVectorizer(max_features=50)
        tfidf_mat = tfidf.fit_transform(X["name"].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(),
            columns=[f"name_tfidf_{i}" for i in range(tfidf_mat.shape[1])],
            index=X.index,
        )
        X = X.drop(columns=["name"])
        X = pd.concat([X, tfidf_df], axis=1)

    # Simple one-hot for object columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X