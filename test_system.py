#!/usr/bin/env python3
"""Safe local smoke tests for EnterpriseOps AI. Does not create tickets/workflows."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FASTAPI = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000").rstrip("/")
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")


def env_value(key, default=None):
    p = Path(".env")
    if p.exists():
        for raw in p.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return default


def call(method, url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, code = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode("utf-8", "replace"), e.code
    elapsed = time.perf_counter() - start
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = raw
    return code, body, elapsed


counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def report(status, name, detail=""):
    counts[status] += 1
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser(description="Run EnterpriseOps AI smoke tests.")
    parser.add_argument("--order", help="Real existing order number, e.g. ORD-1001")
    parser.add_argument(
        "--java-base",
        default=os.getenv("JAVA_BASE_URL") or env_value("JAVA_BASE_URL")
                or env_value("java_base_url") or "http://localhost:8080",
        help="Spring Boot base URL (default http://localhost:8080)",
    )
    args = parser.parse_args()
    java = args.java_base.rstrip("/")

    print("\n=== EnterpriseOps AI smoke tests ===")
    print(f"FastAPI={FASTAPI} | Ollama={OLLAMA} | Model={MODEL} | Spring Boot={java}")
    print("Safe mode: does not create tickets or submit workflows.\n")

    # FastAPI health
    try:
        code, body, sec = call("GET", f"{FASTAPI}/health", timeout=10)
        if code == 200 and isinstance(body, dict) and body.get("status") == "healthy":
            report("PASS", "FastAPI health", f"{sec:.2f}s")
        else:
            report("FAIL", "FastAPI health", f"HTTP {code}: {body}")
    except Exception as e:
        report("FAIL", "FastAPI health", str(e))

    # Ollama + model presence
    model_available = False
    try:
        code, body, sec = call("GET", f"{OLLAMA}/api/tags", timeout=10)
        names = [m.get("name", "") for m in body.get("models", [])] if isinstance(body, dict) else []
        if code == 200:
            report("PASS", "Ollama service", f"{sec:.2f}s")
            model_available = any(n == MODEL or n.startswith(MODEL + ":") for n in names)
            report("PASS" if model_available else "FAIL", "Configured model installed",
                   MODEL if model_available else f"{MODEL} not found; available: {names}")
        else:
            report("FAIL", "Ollama service", f"HTTP {code}: {body}")
    except Exception as e:
        report("FAIL", "Ollama service/model", str(e))