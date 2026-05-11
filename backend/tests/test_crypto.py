"""Phase 3 — Fernet-based BYOK encryption tests."""

from __future__ import annotations

import pytest

from app.crypto import decrypt_api_key, derive_fernet_key, encrypt_api_key


def test_encrypt_decrypt_roundtrip():
    secret = "change-this-32-byte-secret-now!!!"
    plaintext = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA"
    cipher = encrypt_api_key(plaintext, secret)
    assert isinstance(cipher, bytes)
    assert plaintext.encode() not in cipher  # plaintext must not appear verbatim
    recovered = decrypt_api_key(cipher, secret)
    assert recovered == plaintext


def test_decrypt_with_wrong_key_fails():
    secret_a = "secret-A-thirty-two-chars-long-x"
    secret_b = "secret-B-different-and-also-32-x"
    cipher = encrypt_api_key("sk-test-1234", secret_a)
    with pytest.raises(ValueError, match=r"decrypt|invalid|token"):
        decrypt_api_key(cipher, secret_b)


def test_keys_stored_as_bytes_not_str():
    cipher = encrypt_api_key("sk-test", "any-secret-value-32-chars-long-x")
    assert isinstance(cipher, bytes), f"expected bytes, got {type(cipher).__name__}"


def test_derive_fernet_key_is_deterministic():
    secret = "the-same-secret-every-time-please"
    k1 = derive_fernet_key(secret)
    k2 = derive_fernet_key(secret)
    assert k1 == k2, "key derivation must be stable across calls"
    assert isinstance(k1, bytes)
    # Fernet keys are 32 bytes URL-safe base64-encoded → 44 chars (with padding).
    assert len(k1) == 44


def test_derive_fernet_key_differs_per_secret():
    k1 = derive_fernet_key("alpha-secret-thirty-two-chars-xx")
    k2 = derive_fernet_key("beta-secret-thirty-two-chars-xxx")
    assert k1 != k2


def test_encrypt_handles_empty_string():
    """Edge case: an empty plaintext is still valid input — Fernet
    encrypts it deterministically and we recover the empty string."""
    cipher = encrypt_api_key("", "any-secret-value-32-chars-long-x")
    assert decrypt_api_key(cipher, "any-secret-value-32-chars-long-x") == ""


def test_unicode_plaintext_round_trips():
    """Some local providers ship API keys that aren't pure ASCII; the
    helper must encode → decode without dropping bytes."""
    pt = "🔑-token-with-emoji-and-üñîçødé"
    secret = "another-32-char-secret-stringxx"
    cipher = encrypt_api_key(pt, secret)
    assert decrypt_api_key(cipher, secret) == pt
