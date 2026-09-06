"""Temporary loopback-only CONNECT relay through the original project SSH server.

The only permitted destination is the exact Alibaba business host on port 443.
TLS is passed through unchanged; this process never decrypts provider traffic.
No remote command, listener, file write, deployment, or automatic reconnect is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import select
import signal
import socket
import socketserver
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOST = "llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com"
ALLOWED_AUTHORITY = f"{ALLOWED_HOST}:443"
MAX_HEADER_BYTES = 16384
MAX_CONNECTIONS = 8


def request_allowed(header: bytes) -> bool:
    """Only accept a complete HTTP CONNECT request for the exact authority."""
    if len(header) > MAX_HEADER_BYTES or not header.endswith(b"\r\n\r\n"):
        return False
    try:
        lines = header.decode("ascii").split("\r\n")
    except UnicodeDecodeError:
        return False
    request = lines[0].split(" ")
    if request not in (
        ["CONNECT", ALLOWED_AUTHORITY, "HTTP/1.1"],
        ["CONNECT", ALLOWED_AUTHORITY, "HTTP/1.0"],
    ):
        return False
    for line in lines[1:-2]:
        name, separator, value = line.partition(":")
        if not separator or not name or name.strip() != name:
            return False
        if name.lower() == "host" and value.strip() != ALLOWED_AUTHORITY:
            return False
        if name.lower() == "transfer-encoding":
            return False
        if name.lower() == "content-length" and value.strip() != "0":
            return False
    return True


def reject(connection: socket.socket, status: str) -> None:
    try:
        connection.sendall(
            f"HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".encode("ascii")
        )
    except OSError:
        pass


class RelayServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, transport, stop_event: threading.Event,
                 idle_timeout: float, tunnel_timeout: float):
        self.transport = transport
        self.stop_event = stop_event
        self.idle_timeout = idle_timeout
        self.tunnel_timeout = tunnel_timeout
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(("127.0.0.1", port), RelayHandler)
        self.timeout = 0.5

    def process_request(self, request, client_address):
        request.settimeout(5)
        if not self.slots.acquire(blocking=False):
            reject(request, "503 Service Unavailable")
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Never emit tracebacks, CONNECT headers, tunneled bytes, or credentials.
        pass


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        channel = None
        try:
            received = bytearray()
            while b"\r\n\r\n" not in received:
                chunk = self.request.recv(min(4096, MAX_HEADER_BYTES + 1 - len(received)))
                if not chunk:
                    return
                received.extend(chunk)
                if len(received) > MAX_HEADER_BYTES:
                    reject(self.request, "431 Request Header Fields Too Large")
                    return
            raw_header, initial_payload = bytes(received).split(b"\r\n\r\n", 1)
            if not request_allowed(raw_header + b"\r\n\r\n"):
                reject(self.request, "403 Forbidden")
                return
            if self.server.stop_event.is_set() or not self.server.transport.is_active():
                reject(self.request, "503 Service Unavailable")
                return
            try:
                channel = self.server.transport.open_channel(
                    "direct-tcpip", (ALLOWED_HOST, 443), self.client_address, timeout=15
                )
            except Exception:
                reject(self.request, "502 Bad Gateway")
                return
            channel.settimeout(10)
            self.request.settimeout(10)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if initial_payload:
                channel.sendall(initial_payload)
            started = last_activity = time.monotonic()
            while not self.server.stop_event.is_set():
                now = time.monotonic()
                if now - started >= self.server.tunnel_timeout or now - last_activity >= self.server.idle_timeout:
                    break
                readable, _, _ = select.select([self.request, channel], [], [], 1)
                for source in readable:
                    payload = source.recv(65536)
                    if not payload:
                        return
                    destination = channel if source is self.request else self.request
                    destination.sendall(payload)
                    last_activity = time.monotonic()
        except (OSError, EOFError):
            pass
        finally:
            if channel is not None:
                channel.close()


def main() -> int:
    import paramiko
    from probe_server_provider_tls import PROJECT_SERVER, load_project_credentials

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--credentials-file", type=Path, default=PROJECT_ROOT.parent / "服务器信息.txt")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh" / "known_hosts")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--max-lifetime", type=float, default=14400)
    parser.add_argument("--idle-timeout", type=float, default=180)
    parser.add_argument("--tunnel-timeout", type=float, default=900)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or min(args.max_lifetime, args.idle_timeout, args.tunnel_timeout) <= 0:
        parser.error("port or timeouts out of range")

    # Library diagnostics are suppressed, including unexpected SSH exception bodies.
    logging.getLogger("paramiko").disabled = True
    logging.getLogger("paramiko.transport").disabled = True
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_args: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_args: stop_event.set())
    report = {
        "schemaVersion": 1,
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "listenHost": "127.0.0.1",
        "listenPort": None,
        "allowedAuthority": ALLOWED_AUTHORITY,
        "sshServer": PROJECT_SERVER,
        "sshAuthenticationAttempts": 0,
        "sshAuthenticated": False,
        "sshTcpForwardAllowed": False,
        "maxConcurrentTunnels": MAX_CONNECTIONS,
        "maxLifetimeSeconds": args.max_lifetime,
        "idleTimeoutSeconds": args.idle_timeout,
        "tunnelTimeoutSeconds": args.tunnel_timeout,
        "tlsTerminated": False,
        "remoteMutationPerformed": False,
        "automaticReconnect": False,
        "status": "starting",
    }

    def emit_status():
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
        args.status_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)

    client = paramiko.SSHClient()
    server = None
    exit_code = 2
    try:
        if args.stop_file.exists():
            raise RuntimeError("stop requested before startup")
        client.load_host_keys(str(args.known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        username, password = load_project_credentials(args.credentials_file)
        report["sshAuthenticationAttempts"] = 1
        client.connect(PROJECT_SERVER, username=username, password=password,
                       allow_agent=False, look_for_keys=False, timeout=10,
                       banner_timeout=15, auth_timeout=20)
        password = ""
        report["sshAuthenticated"] = True
        transport = client.get_transport()
        transport.set_keepalive(30)
        preflight = transport.open_channel("direct-tcpip", (ALLOWED_HOST, 443), ("127.0.0.1", 0), timeout=15)
        preflight.close()
        report["sshTcpForwardAllowed"] = True
        server = RelayServer(args.port, transport, stop_event, args.idle_timeout, args.tunnel_timeout)
        report["listenPort"] = server.server_address[1]
        report["status"] = "ready"
        emit_status()
        started = time.monotonic()
        while not stop_event.is_set():
            if args.stop_file.exists():
                report["stopReason"] = "stop_file"
                break
            if time.monotonic() - started >= args.max_lifetime:
                report["stopReason"] = "max_lifetime"
                break
            if not transport.is_active():
                report["stopReason"] = "ssh_disconnected"
                break
            server.handle_request()
        report["status"] = "stopped"
        report.setdefault("stopReason", "signal")
        exit_code = 0
    except Exception as error:
        report["status"] = "failed"
        report["errorType"] = type(error).__name__
    finally:
        stop_event.set()
        if server is not None:
            server.server_close()
        client.close()
        report["stoppedAt"] = datetime.now(timezone.utc).isoformat()
        emit_status()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
