"""BYOK API-key encryption.

Stores user-supplied LLM API keys at rest using Fernet (AES-128-CBC +
HMAC-SHA-256). The Fernet key itself is derived from the platform-wide
``KEY_ENCRYPTION_SECRET`` via SHA-256 → URL-safe base64 — deterministic,
no per-key salt, so a single secret in the operator's environment
suffices to decrypt every stored API key.

The deterministic derivation is a deliberate choice: the BYOK threat
model is "attacker dumps the SQLite database" not "attacker reads the
operator's environment." Per-key salts would defeat the former without
helping with the latter, and would force every database read to also
load a salt blob. Any rotation of the secret is a planned operator
action (re-encrypt-all migration), not an automatic per-row event.

Public API
----------
``derive_fernet_key(secret)``      → bytes  (44-byte URL-safe b64)
``encrypt_api_key(plaintext, secret)`` → bytes  (Fernet token)
``decrypt_api_key(token, secret)``  → str   (raises ValueError on bad token)
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


def derive_fernet_key(secret: str) -> bytes:
    """Map an arbitrary-length secret string to a Fernet-compatible key.

    SHA-256 the secret → 32 raw bytes → URL-safe base64 → 44 ASCII bytes
    (Fernet's required key format). Deterministic: the same secret
    always yields the same key, so the operator can decrypt records
    written by any prior boot of the platform.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_api_key(plaintext: str, secret: str) -> bytes:
    """Encrypt ``plaintext`` under the platform secret. Returns the
    Fernet token as ``bytes`` (never as ``str``) so callers don't
    accidentally store text where a BLOB is required."""
    fernet = Fernet(derive_fernet_key(secret))
    return fernet.encrypt(plaintext.encode("utf-8"))


def decrypt_api_key(token: bytes, secret: str) -> str:
    """Decrypt a Fernet token. Raises ``ValueError`` (with the underlying
    Fernet detail wrapped) on a bad key, tampered ciphertext, or wrong
    secret — Fernet's authenticator catches all three."""
    fernet = Fernet(derive_fernet_key(secret))
    try:
        plain = fernet.decrypt(token)
    except InvalidToken as e:
        raise ValueError("invalid token: decrypt failed (wrong secret or tampered)") from e
    return plain.decode("utf-8")
