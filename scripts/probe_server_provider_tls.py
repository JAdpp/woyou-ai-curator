"""Run three credential-free HTTPS probes through the project's existing SSH.

The SSH password is loaded only in memory from the original server information
document. Host keys must already be trusted. No deployment, remote files,
services, environment reads or SSH authorization changes are performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SERVER = "47.89.246.208"
PROVIDERS = (
    ("aliyun_business", "https://llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com/"),
    ("dashscope", "https://dashscope.aliyuncs.com/"),
    ("deepseek", "https://api.deepseek.com/"),
)


def load_project_credentials(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    if PROJECT_SERVER not in text:
        raise ValueError("document does not identify the original project server")
    usernames = []
    passwords = []
    for line in text.splitlines():
        parts = re.split(r"[:：=]", line, maxsplit=1)
        if len(parts) != 2:
            continue
        label, value = parts[0].strip(), parts[1].strip()
        if re.search(r"密码|password", label, re.IGNORECASE):
            passwords.append(value)
        elif re.search(r"用户|账户|帐号|账号|username|user", label, re.IGNORECASE):
            usernames.append(value)
    if len(usernames) != 1 or len(passwords) != 1 or not usernames[0] or not passwords[0]:
        raise ValueError("document must contain one username and one password field")
    return usernames[0], passwords[0]


def probe(client: paramiko.SSHClient, name: str, url: str) -> dict:
    # HEAD requests discard response headers and never carry provider keys.
    command = " ".join(shlex.quote(value) for value in (
        "curl", "--noproxy", "*", "--head", "--silent", "--connect-timeout", "8",
        "--max-time", "18", "--output", "/dev/null", "--write-out",
        "HTTP:%{http_code} VERIFY:%{ssl_verify_result}", url,
    ))
    result = {"provider": name, "url": url, "method": "HEAD", "executedFromServer": False,
              "apiKeyUsed": False, "httpStatus": None, "tlsResult": "unknown", "curlExitCode": None}
    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=25)
        result["executedFromServer"] = True
        output = stdout.read(512).decode("ascii", errors="replace")
        stderr.read(4096)  # Never copy arbitrary remote output into the report.
        result["curlExitCode"] = stdout.channel.recv_exit_status()
        match = re.fullmatch(r"HTTP:(\d{3}) VERIFY:(\d+)", output.strip())
        if match:
            status, verify = int(match.group(1)), int(match.group(2))
            result["httpStatus"] = status or None
            result["tlsResult"] = "passed" if status > 0 and verify == 0 else "failed_or_not_reached"
            result["certificateVerificationCode"] = verify
        else:
            result["tlsResult"] = "probe_output_unavailable"
    except Exception as error:
        result["errorType"] = type(error).__name__
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, default=PROJECT_ROOT.parent / "服务器信息.txt")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh" / "known_hosts")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/qa/rag-optimization-20260906/network-server-probe.json")
    args = parser.parse_args()
    report = {"schemaVersion": 2, "probedAt": datetime.now(timezone.utc).isoformat(),
              "server": PROJECT_SERVER, "authentication": {"attempts": 0, "success": False},
              "providerProbes": [], "remoteMutationPerformed": False}
    client = paramiko.SSHClient()
    try:
        client.load_host_keys(str(args.known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        username, password = load_project_credentials(args.credentials_file)
        report["authentication"]["attempts"] = 1
        client.connect(PROJECT_SERVER, username=username, password=password,
                       allow_agent=False, look_for_keys=False, timeout=10,
                       banner_timeout=15, auth_timeout=20)
        password = ""
        report["authentication"]["success"] = True
        report["providerProbes"] = [probe(client, name, url) for name, url in PROVIDERS]
    except Exception as error:
        # Exception bodies may contain library request details; record only type.
        report["authentication"]["errorType"] = type(error).__name__
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["authentication"]["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
