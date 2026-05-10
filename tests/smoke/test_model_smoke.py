import numpy as np
import pandas as pd

from training.helpers import encode_dataframe, train


def test_predict_shape_and_probabilities():
    df = encode_dataframe(pd.read_csv("data/tennis.csv"))
    model, _ = train(df)
    X = df[["outlook", "temperature", "humidity", "wind"]]
    proba = model.predict_proba(X)
    assert proba.shape == (14, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
