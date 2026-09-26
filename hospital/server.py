"""Threaded TCP server that receives patient vitals from ambulances."""
import socketserver
import threading

from common.protocol import HOST, PORT, recv_message, send_message


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        srv = self.server
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        try:
            msg = recv_message(self.request)
            if msg.get("type") != "patientdata":
                raise ValueError(f"unknown message type {msg.get('type')!r}")
            features = msg["features"]
            label, proba, model_name = srv.predictor(features)
            reply = {"ok": True, "condition": label, "probability": proba, "model": model_name}
        except Exception as exc:  # report any problem back to the ambulance
            features = None
            reply = {"ok": False, "error": str(exc)}
        try:
            send_message(self.request, reply)
        finally:
            srv.on_event(peer, features, reply)


class _TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class PredictionServer:
    """Wraps the socket server so the GUI can start/stop it safely.

    predictor(features) -> (label, probability, model_name)
    on_event(peer, features, reply) is called from worker threads.
    """

    def __init__(self, predictor, on_event=lambda *a: None, host=HOST, port=PORT):
        self.predictor = predictor
        self.on_event = on_event
        self.host, self.port = host, port
        self._server = None
        self._thread = None

    @property
    def running(self):
        return self._server is not None

    def start(self):
        if self.running:
            return
        self._server = _TCPServer((self.host, self.port), _Handler)
        self._server.predictor = self.predictor
        self._server.on_event = self.on_event
        self.port = self._server.server_address[1]  # resolves port 0 in tests
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
