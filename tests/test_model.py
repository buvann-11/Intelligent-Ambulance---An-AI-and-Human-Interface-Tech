from pathlib import Path

import pytest

from hospital.model import ALGORITHMS, HeartModel, load_dataset

DATA = Path(__file__).resolve().parents[1] / "hospital" / "data" / "heart.csv"


@pytest.fixture(scope="module")
def model():
    return HeartModel(load_dataset(DATA))


def test_duplicates_removed():
    assert len(load_dataset(DATA)) < len(load_dataset(DATA, drop_duplicates=False))


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_every_algorithm_trains(model, name):
    m = model.train(name)
    assert 50 <= m.accuracy <= 100
    assert m.confusion.shape == (2, 2)


def test_predict_uses_best_model(model):
    for name in ALGORITHMS:
        model.train(name)
    label, proba, used = model.predict([65, 0, 2, 160, 360, 0, 0, 151, 0, 0.8, 2, 0, 2])
    assert label in ("Normal", "Abnormal")
    assert used == model.best_model_name()
    assert proba is None or 0 <= proba <= 1


def test_predict_rejects_wrong_length(model):
    model.train("KNN")
    with pytest.raises(ValueError):
        model.predict([1, 2, 3])
