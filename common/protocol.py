"""Wire protocol shared by the ambulance client and the hospital server.

Messages are UTF-8 JSON objects, one per line. JSON is used instead of
pickle because unpickling data from the network can execute arbitrary code.
"""
import json
import os
import socket

HOST = os.environ.get("AMBULANCE_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMBULANCE_SERVER_PORT", "2222"))

# Feature order expected by the model (UCI / Kaggle heart disease dataset).
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MAX_MESSAGE_BYTES = 64 * 1024


def send_message(sock: socket.socket, message: dict) -> None:
    sock.sendall((json.dumps(message) + "\n").encode("utf-8"))


def recv_message(sock: socket.socket) -> dict:
    """Read one newline-terminated JSON message from the socket."""
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed before a full message arrived")
        buf.extend(chunk)
        if len(buf) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
    return json.loads(buf.decode("utf-8"))


def request_prediction(features, host: str = HOST, port: int = PORT, timeout: float = 5.0) -> dict:
    """Send one patient's vitals to the hospital and return its reply."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_message(sock, {"type": "patientdata", "features": [float(x) for x in features]})
        return recv_message(sock)
