"""t3_login.py — Obtain a T3 (Triticeae Toolbox) BrAPI access token.

T3 authenticates via a username/password POST to the BrAPI token endpoint
(BrAPI.R tutorial:
https://github.com/TriticeaeToolbox/BrAPI.R/blob/main/TUTORIAL.md).  The
token is short-lived (~2 h) and is stored to a local file so that
t3_brapi_export.py can reuse it.

Password is read with getpass (no echo) and never printed or stored.

Usage:
    python scripts/t3_login.py --username you@example.com
    # or, non-interactively (password via env — avoid putting it in shell history):
    set T3_PASSWORD=... & python scripts/t3_login.py --username you@example.com
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_TOKEN_FILE = Path("data/t3/.t3_token")
TOKEN_ENDPOINT_VERSION = "v1"  # v1 returns a clean {access_token, expires_in}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="https://wheat.triticeaetoolbox.org/",
                        help="T3 BrAPI base URL (default wheat)")
    parser.add_argument("--username", default=None,
                        help="T3 username (or set T3_USERNAME); prompts if absent")
    parser.add_argument("--password", default=None,
                        help="T3 password (or set T3_PASSWORD); prompts if absent")
    parser.add_argument("--out", type=Path, default=DEFAULT_TOKEN_FILE,
                        help="Where to write the token (default data/t3/.t3_token)")
    return parser.parse_args(argv)


def fetch_token(base_url: str, username: str, password: str) -> dict:
    url = base_url.rstrip("/") + f"/brapi/{TOKEN_ENDPOINT_VERSION}/token"
    body = urllib.parse.urlencode(
        {"username": username, "password": password}
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_access_token(payload: dict) -> str | None:
    token = payload.get("access_token")
    if token:
        return str(token)
    # Some v2 deployments nest it under result.
    result = payload.get("result", {})
    if isinstance(result, dict):
        token = result.get("access_token")
        if token:
            return str(token)
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    username = args.username or os.environ.get("T3_USERNAME")
    password = args.password or os.environ.get("T3_PASSWORD")
    if not username:
        username = input("T3 username: ").strip()
    if not password:
        password = getpass.getpass("T3 password: ")

    try:
        payload = fetch_token(args.base_url, username, password)
    except urllib.error.HTTPError as e:
        print(f"[t3_login] HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[t3_login] error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    token = _extract_access_token(payload)
    if not token:
        statuses = payload.get("metadata", {}).get("status", [])
        messages = [s.get("message", "") for s in statuses if isinstance(s, dict)]
        print(f"[t3_login] no access_token in response: {'; '.join(messages) or payload}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(token, encoding="utf-8")
    print(f"[t3_login] token obtained (expires ~{payload.get('expires_in', '?')}s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
