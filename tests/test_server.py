"""End-to-end: ambulance client -> hospital server over a real socket."""
from pathlib import Path

import pandas as pd

from common.protocol import FEATURES, request_prediction
from hospital.model import HeartModel, load_dataset
from hospital.server import PredictionServer

ROOT = Path(__file__).resolve().parents[1]


def test_round_trip():
    model = HeartModel(load_dataset(ROOT / "hospital" / "data" / "heart.csv"))
    model.train("Random Forest")
    events = []
    server = PredictionServer(model.predict, lambda *e: events.append(e), host="127.0.0.1", port=0)
    server.start()
    try:
        rows = pd.read_csv(ROOT / "ambulance" / "data" / "test_patients.csv")[FEATURES].head(5)
        for row in rows.to_numpy().tolist():
            reply = request_prediction(row, port=server.port)
            assert reply["ok"], reply
            assert reply["condition"] in ("Normal", "Abnormal")
        bad = request_prediction([1, 2], port=server.port)
        assert not bad["ok"]
    finally:
        server.stop()
    assert len(events) == 6
