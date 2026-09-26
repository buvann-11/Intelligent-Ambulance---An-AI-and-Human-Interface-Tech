"""Ambulance side: stream patient vitals to the hospital and show its assessment.

Run from the repository root:  python -m ambulance.app
"""
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import pandas as pd

from common.protocol import FEATURES, HOST, PORT, request_prediction

DATA_DIR = Path(__file__).resolve().parent / "data"
SEND_INTERVAL_SECONDS = 1.0


class AmbulanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        root.title("Intelligent Ambulance – Ambulance Unit")
        root.geometry("950x560")
        root.minsize(700, 420)
        root.configure(bg="turquoise")
        root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(root, text="Ambulance Application", bg="dark goldenrod", fg="white",
                 font=("Times", 16, "bold"), height=2).pack(fill="x")

        self.text = scrolledtext.ScrolledText(root, font=("Consolas", 11), wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

        bar = tk.Frame(root, bg="turquoise")
        bar.pack(pady=(0, 10))
        self.send_btn = tk.Button(bar, text="Report Patient Condition to Hospital Server",
                                  font=("Times", 13, "bold"), command=self.start)
        self.send_btn.pack(side="left", padx=5)
        self.stop_btn = tk.Button(bar, text="Stop", font=("Times", 13, "bold"),
                                  command=self.stop_flag.set, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.text.insert("end", f"Hospital server: {HOST}:{PORT}\n"
                                "Start the hospital app, train a model and start its server first.\n\n")
        self.root.after(100, self._drain_events)

    def start(self):
        path = filedialog.askopenfilename(initialdir=DATA_DIR, title="Choose patient readings",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path).fillna(0)
            rows = df[FEATURES].to_numpy(dtype=float).tolist()
        except KeyError as exc:
            messagebox.showerror("Wrong file", f"Missing columns: {exc}")
            return
        except Exception as exc:
            messagebox.showerror("Could not read file", str(exc))
            return
        self.stop_flag.clear()
        self.send_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.worker = threading.Thread(target=self._send_all, args=(rows,), daemon=True)
        self.worker.start()

    def _send_all(self, rows):  # runs in a background thread
        for i, row in enumerate(rows, 1):
            if self.stop_flag.is_set():
                break
            try:
                reply = request_prediction(row)
            except OSError as exc:
                self.events.put(("error", f"Cannot reach hospital at {HOST}:{PORT} ({exc}). "
                                          "Is its server started?"))
                break
            self.events.put(("reply", (i, row, reply)))
            time.sleep(SEND_INTERVAL_SECONDS)
        self.events.put(("done", None))

    def _drain_events(self):
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "reply":
                i, row, reply = payload
                vitals = ",".join(f"{v:g}" for v in row)
                if reply.get("ok"):
                    p = reply.get("probability")
                    risk = f" (risk {p:.0%})" if p is not None else ""
                    self.text.insert("end", f"Patient {i}: {vitals}\n  ==> Condition {reply['condition']}{risk}\n\n")
                else:
                    self.text.insert("end", f"Patient {i}: hospital error – {reply.get('error')}\n\n")
            elif kind == "error":
                self.text.insert("end", payload + "\n\n")
            elif kind == "done":
                self.send_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
            self.text.see("end")
        self.root.after(100, self._drain_events)

    def close(self):
        self.stop_flag.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    AmbulanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
