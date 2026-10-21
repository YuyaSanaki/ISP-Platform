"""Optional login for the Web UI.

Off unless ``ISP_WEBUI_USERS_FILE`` names a YAML file of users and password hashes::

    users:
      alice: pbkdf2_sha256$600000$<salt hex>$<hash hex>

Create an entry with ``python webui/webui_auth.py hash alice`` (prompts for the password).
With login on, every page needs a user name and password, and the ortholog approval card
records the logged-in user instead of a free-text name. Without it the Web UI is meant for
a single trusted user on localhost or behind an access-controlled proxy.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path
from typing import Mapping

import yaml

USERS_FILE_ENV = "ISP_WEBUI_USERS_FILE"
_SCHEME = "pbkdf2_sha256"
_ITERATIONS = 600_000


class AuthConfigError(ValueError):
    """The users file is missing or malformed."""


def hash_password(password: str, *, iterations: int = _ITERATIONS, salt: bytes | None = None) -> str:
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = str(encoded).split("$")
        if scheme != _SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def users_file() -> Path | None:
    raw = os.environ.get(USERS_FILE_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def load_users(path: Path) -> dict[str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as e:
        raise AuthConfigError(f"{USERS_FILE_ENV}={path}: cannot read ({e})") from e
    users = data.get("users") if isinstance(data, Mapping) else None
    if not isinstance(users, Mapping) or not users:
        raise AuthConfigError(f"{USERS_FILE_ENV}={path}: needs a non-empty 'users' mapping")
    return {str(k): str(v) for k, v in users.items()}


def check_login(users: Mapping[str, str], user: str, password: str) -> bool:
    encoded = users.get(str(user or "").strip())
    if encoded is None:
        # Same work for unknown users, so timing does not reveal valid names.
        verify_password(password, hash_password("x", iterations=_ITERATIONS, salt=b"\0" * 16))
        return False
    return verify_password(password, encoded)


def require_login(st) -> str | None:
    """Stop the page until the user logs in. Returns the user name, or None when login is off."""
    path = users_file()
    if path is None:
        return None
    user = st.session_state.get("webui_auth_user")
    if user:
        return str(user)
    try:
        users = load_users(path)
    except AuthConfigError as e:
        st.error(f"Login is configured but cannot be used: {e}")
        st.stop()
    st.subheader("Log in")
    with st.form("webui_login"):
        name = st.text_input("User name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if check_login(users, name, password):
            st.session_state["webui_auth_user"] = name.strip()
            st.rerun()
        st.error("Wrong user name or password.")
    st.stop()
    return None


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "hash":
        print("usage: python webui/webui_auth.py hash <user>", file=sys.stderr)
        return 2
    password = getpass.getpass(f"Password for {argv[1]}: ")
    if password != getpass.getpass("Repeat: "):
        print("Passwords differ.", file=sys.stderr)
        return 1
    print(yaml.safe_dump({"users": {argv[1]: hash_password(password)}}, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
