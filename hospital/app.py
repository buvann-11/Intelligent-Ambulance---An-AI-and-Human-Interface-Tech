"""Hospital side: train heart-condition models and serve predictions to ambulances.

Run from the repository root:  python -m hospital.app
"""
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.protocol import HOST, PORT
from hospital.model import ALGORITHMS, LABELS, HeartModel, load_dataset
from hospital.server import PredictionServer

DATA_DIR = Path(__file__).resolve().parent / "data"


class HospitalApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.dataset = None
        self.model = None
        self.events = queue.Queue()  # worker threads -> GUI thread
        self.server = PredictionServer(self._predict, self._on_request)

        root.title("Intelligent Ambulance – Hospital Server")
        root.geometry("1150x680")
        root.minsize(900, 560)
        root.configure(bg="turquoise")
        root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(root, text="Intelligent Ambulance – AI and Human Interface Technology",
                 bg="dark goldenrod", fg="white", font=("Times", 16, "bold"),
                 height=2).pack(fill="x")

        body = tk.Frame(root, bg="turquoise")
        body.pack(fill="both", expand=True, padx=10, pady=10)

        self.text = scrolledtext.ScrolledText(body, font=("Consolas", 11), wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(body, bg="turquoise")
        panel.pack(side="right", fill="y", padx=(10, 0))
        btn = dict(font=("Times", 12, "bold"), width=34, pady=4)

        tk.Button(panel, text="1. Upload Heart Disease Dataset", command=self.upload, **btn).pack(pady=4)
        self.dedup = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="Remove duplicate rows (recommended)", variable=self.dedup,
                       bg="turquoise", font=("Times", 11)).pack(anchor="w")
        tk.Button(panel, text="2. Preprocess & Train/Test Split", command=self.preprocess, **btn).pack(pady=4)
        for name in ALGORITHMS:
            tk.Button(panel, text=f"3. Run {name}", command=lambda n=name: self.run(n), **btn).pack(pady=4)
        tk.Button(panel, text="Run All Algorithms", command=self.run_all, **btn).pack(pady=4)
        tk.Button(panel, text="Comparison Graph", command=self.graph, **btn).pack(pady=4)

        tk.Label(panel, text="Model used for live predictions:", bg="turquoise",
                 font=("Times", 11, "bold")).pack(anchor="w", pady=(12, 0))
        self.serving = tk.StringVar(value="Best (highest F1)")
        self.serving_box = ttk.Combobox(panel, textvariable=self.serving, state="readonly",
                                        values=["Best (highest F1)"], width=36)
        self.serving_box.pack(pady=2)

        self.server_btn = tk.Button(panel, text="4. Start Receiving Patient Data",
                                    command=self.toggle_server, **btn)
        self.server_btn.pack(pady=(10, 4))
        self.status = tk.Label(panel, text="Server stopped", bg="turquoise", fg="firebrick",
                               font=("Times", 11, "bold"))
        self.status.pack()

        self.root.after(100, self._drain_events)

    # ------------------------------------------------------------------ helpers
    def log(self, msg="", clear=False):
        if clear:
            self.text.delete("1.0", "end")
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def _need(self, what):
        if what == "dataset" and self.dataset is None:
            messagebox.showwarning("Dataset needed", "Upload a dataset first (step 1).")
            return False
        if what == "model" and self.model is None:
            messagebox.showwarning("Preprocess first", "Run preprocessing & split first (step 2).")
            return False
        return True

    # ------------------------------------------------------------------ steps
    def upload(self):
        path = filedialog.askopenfilename(initialdir=DATA_DIR, title="Choose heart.csv",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            raw = pd.read_csv(path)
            self.dataset = load_dataset(path, drop_duplicates=self.dedup.get())
        except Exception as exc:
            messagebox.showerror("Could not load dataset", str(exc))
            return
        self.model = None
        self.log(f"{path} loaded", clear=True)
        self.log(str(self.dataset.head()) + "\n")
        self.log(f"Rows in file            : {len(raw)}")
        self.log(f"Rows used (after dedup) : {len(self.dataset)}")
        self.log(f"Attributes              : {self.dataset.shape[1] - 1} features + target")
        counts = self.dataset["target"].value_counts().sort_index()
        self.log("Class balance           : " +
                 ", ".join(f"{LABELS[k]}={v}" for k, v in counts.items()))
        plt.figure("Class distribution")
        counts.rename(index=dict(enumerate(LABELS))).plot(kind="bar", color=["seagreen", "indianred"])
        plt.title("Patients per class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show(block=False)

    def preprocess(self):
        if not self._need("dataset"):
            return
        self.model = HeartModel(self.dataset)
        self._refresh_serving()
        self.log("Preprocessing complete: stratified 80/20 split, StandardScaler fitted on training data only.",
                 clear=True)
        self.log(f"Training records : {len(self.model.y_train)}")
        self.log(f"Testing records  : {len(self.model.y_test)}")

    def run(self, name, show_plot=True):
        if not self._need("model"):
            return
        m = self.model.train(name)
        self.log(f"\n{name}")
        for k, v in m.as_dict().items():
            self.log(f"  {k:<9}: {v:6.2f}%")
        self._refresh_serving()
        if show_plot:
            plt.figure(f"{name} confusion matrix")
            sns.heatmap(m.confusion, xticklabels=LABELS, yticklabels=LABELS,
                        annot=True, cmap="viridis", fmt="g")
            plt.title(f"{name} Confusion Matrix")
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.tight_layout()
            plt.show(block=False)

    def run_all(self):
        if not self._need("model"):
            return
        for name in ALGORITHMS:
            self.run(name, show_plot=False)
        self.log(f"\nBest model: {self.model.best_model_name()}")

    def graph(self):
        if self.model is None or not self.model.metrics:
            messagebox.showwarning("No results", "Run at least one algorithm first.")
            return
        df = pd.DataFrame({n: m.as_dict() for n, m in self.model.metrics.items()})
        ax = df.plot(kind="bar", figsize=(9, 5), rot=0)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title("Algorithm comparison")
        plt.tight_layout()
        plt.show(block=False)

    def _refresh_serving(self):
        names = list(self.model.metrics) if self.model else []
        self.serving_box["values"] = ["Best (highest F1)"] + names
        if self.serving.get() not in self.serving_box["values"]:
            self.serving.set("Best (highest F1)")

    # ------------------------------------------------------------------ server
    def toggle_server(self):
        if self.server.running:
            self.server.stop()
            self.server_btn.config(text="4. Start Receiving Patient Data")
            self.status.config(text="Server stopped", fg="firebrick")
            return
        if self.model is None or not self.model.models:
            messagebox.showwarning("No model", "Train at least one algorithm before starting the server.")
            return
        try:
            self.server.start()
        except OSError as exc:
            messagebox.showerror("Could not start server", f"{HOST}:{PORT} – {exc}")
            return
        self.server_btn.config(text="Stop Server")
        self.status.config(text=f"Listening on {HOST}:{self.server.port}", fg="darkgreen")
        self.log(f"\nHospital server started on {HOST}:{self.server.port}. Waiting for ambulances...")

    def _predict(self, features):  # called from worker threads
        choice = self.serving.get()
        name = None if choice.startswith("Best") else choice
        return self.model.predict(features, name)

    def _on_request(self, peer, features, reply):  # called from worker threads
        self.events.put((peer, features, reply))

    def _drain_events(self):
        while not self.events.empty():
            peer, features, reply = self.events.get()
            if reply["ok"]:
                p = reply["probability"]
                prob = f" (risk {p:.0%})" if p is not None else ""
                vitals = ",".join(f"{v:g}" for v in features)
                self.log(f"[{peer}] {vitals} ==> {reply['condition']}{prob} via {reply['model']}")
            else:
                self.log(f"[{peer}] rejected: {reply['error']}")
        self.root.after(100, self._drain_events)

    def close(self):
        self.server.stop()
        plt.close("all")
        self.root.destroy()


def main():
    root = tk.Tk()
    HospitalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
