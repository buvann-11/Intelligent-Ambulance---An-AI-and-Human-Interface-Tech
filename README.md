# Intelligent Ambulance – AI and Human Interface Technology

[![Tests](https://github.com/buvann-11/Intelligent-Ambulance---An-AI-and-Human-Interface-Tech/actions/workflows/tests.yml/badge.svg)](https://github.com/buvann-11/Intelligent-Ambulance---An-AI-and-Human-Interface-Tech/actions/workflows/tests.yml)

An ambulance crew sends a patient's vital signs to the hospital **while still on the road**. The hospital runs a machine-learning model on the readings right away and sends back whether the patient's heart condition looks **Normal** or **Abnormal**, so the emergency team can get ready before the patient arrives.

The project has two desktop apps (Tkinter) that talk over TCP:

```
┌──────────────────────┐   vitals (JSON over TCP)   ┌──────────────────────────────┐
│  Ambulance unit      │ ─────────────────────────▶ │  Hospital server             │
│  ambulance/app.py    │                            │  hospital/app.py             │
│  • reads patient CSV │ ◀───────────────────────── │  • trains DT / RF / KNN      │
│  • shows assessment  │   condition + risk score   │  • serves live predictions   │
└──────────────────────┘                            └──────────────────────────────┘
```

## Features

- **Three classifiers** – Decision Tree, Random Forest and K-Nearest Neighbours, with accuracy, precision, recall, F1 and a confusion matrix for each.
- **Comparison graph** of all trained models.
- **Live prediction server** – multi-threaded; uses the best model (highest F1) or one you pick, and returns a risk probability.
- **Responsive GUIs** – networking runs on background threads, so the windows never freeze. The ambulance can stop a transmission part-way through.
- **Safe protocol** – newline-delimited JSON. The original version used `pickle`, which can run arbitrary code when it loads data from the network.
- **Honest evaluation** – duplicate rows are removed, the split is stratified, and the scaler is fitted on training data only (see [Results](#results)).
- **Automated tests** for the model and the full client ↔ server round trip.

## Dataset

`hospital/data/heart.csv` is the public UCI Cleveland heart disease dataset (1,025 rows, 13 features + `target`).

| Feature | Meaning |
| --- | --- |
| age, sex | Patient demographics |
| cp | Chest-pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak, slope | ST depression and its slope |
| ca | Number of major vessels coloured by fluoroscopy |
| thal | Thalassemia type |
| **target** | 0 = Normal, 1 = Abnormal |

`ambulance/data/test_patients.csv` has 56 unlabeled readings that the ambulance sends to the hospital.

## Getting started

Requires Python 3.9+ with Tkinter. Tkinter comes with the python.org installers. On Debian/Ubuntu, run `sudo apt install python3-tk`.

```bash
git clone https://github.com/buvann-11/Intelligent-Ambulance---An-AI-and-Human-Interface-Tech.git
cd Intelligent-Ambulance---An-AI-and-Human-Interface-Tech
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Run the demo

1. **Start the hospital:** `python -m hospital.app` (or double-click `run_hospital.bat` / run `sh run_hospital.sh`).
   1. *Upload Heart Disease Dataset* → pick `hospital/data/heart.csv`
   2. *Preprocess & Train/Test Split*
   3. *Run All Algorithms* (or run each one to see its confusion matrix), then *Comparison Graph*
   4. *Start Receiving Patient Data*
2. **Start the ambulance** in a second terminal: `python -m ambulance.app` (or `run_ambulance.bat` / `sh run_ambulance.sh`).
   - Click *Report Patient Condition to Hospital Server* → pick `ambulance/data/test_patients.csv`.

The two apps connect on `127.0.0.1:2222` by default. To run them on different machines, set these environment variables on both sides:

```bash
export AMBULANCE_SERVER_HOST=192.168.1.20   # hospital machine's IP
export AMBULANCE_SERVER_PORT=2222
```

## Results

Test-set scores (stratified 80/20 split, `random_state=42`):

| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| Decision Tree | 80.3% | 80.2% |
| KNN (k=10) | 80.3% | 80.2% |
| Random Forest | 75.4% | 75.2% |

> **Why not ~100%?** 723 of the 1,025 rows in `heart.csv` are exact duplicates, which leaves only 302 unique patients. If the duplicates stay in, copies of the same patient end up in both the training set and the test set, and Random Forest scores a misleading **100%**. The *Remove duplicate rows* checkbox (on by default) prevents this. Untick it to see the difference.

## Project structure

```
├── ambulance/
│   ├── app.py               # Ambulance GUI – sends vitals, shows replies
│   └── data/test_patients.csv
├── hospital/
│   ├── app.py               # Hospital GUI – train, evaluate, serve
│   ├── model.py             # Data loading, training, metrics, prediction
│   ├── server.py            # Threaded TCP prediction server
│   └── data/heart.csv
├── common/protocol.py       # JSON wire protocol, host/port, feature order
├── tests/                   # pytest: model + end-to-end socket test
├── requirements.txt
└── run_*.bat / run_*.sh     # One-click launchers
```

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Disclaimer

This is an academic prototype. It is **not** a medical device, and its predictions must not be used for real clinical decisions.

## License

MIT © buvann-11
