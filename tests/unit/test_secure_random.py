"""Tests for SecureRandom utility."""

import string

from src.utils.secure_random import (
    SecureRandom,
    generate_session_id,
    generate_api_key,
    generate_csrf_token,
    generate_oauth_state,
)


class TestSecureRandomTokens:
    def test_token_hex_length(self):
        token = SecureRandom.token_hex(16)
        assert len(token) == 32  # 16 bytes = 32 hex chars
        assert all(c in "0123456789abcdef" for c in token)

    def test_token_hex_default(self):
        token = SecureRandom.token_hex()
        assert len(token) == 64  # 32 bytes default

    def test_token_urlsafe(self):
        token = SecureRandom.token_urlsafe(16)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_bytes(self):
        data = SecureRandom.token_bytes(16)
        assert isinstance(data, bytes)
        assert len(data) == 16

    def test_tokens_are_unique(self):
        tokens = {SecureRandom.token_hex(16) for _ in range(100)}
        assert len(tokens) == 100  # All unique


class TestSecureRandomNumbers:
    def test_randbelow_range(self):
        for _ in range(100):
            val = SecureRandom.randbelow(10)
            assert 0 <= val < 10

    def test_choice(self):
        options = ["a", "b", "c"]
        for _ in range(50):
            assert SecureRandom.choice(options) in options


class TestSecureRandomConvenience:
    def test_session_token(self):
        token = SecureRandom.session_token()
        assert len(token) == 64  # 32 bytes = 64 hex chars

    def test_api_key(self):
        key = SecureRandom.api_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_nonce(self):
        nonce = SecureRandom.nonce()
        assert len(nonce) == 32  # 16 bytes = 32 hex chars


class TestSecureRandomPassword:
    def test_password_length(self):
        pwd = SecureRandom.password(length=20)
        assert len(pwd) == 20

    def test_password_contains_required_chars(self):
        pwd = SecureRandom.password(
            length=16,
            use_digits=True,
            use_uppercase=True,
            use_lowercase=True,
            use_punctuation=True,
        )
        assert len(pwd) == 16
        assert any(c in string.digits for c in pwd)
        assert any(c in string.ascii_uppercase for c in pwd)
        assert any(c in string.ascii_lowercase for c in pwd)
        assert any(c in string.punctuation for c in pwd)

    def test_password_no_punctuation(self):
        pwd = SecureRandom.password(length=16, use_punctuation=False)
        assert not any(c in string.punctuation for c in pwd)

    def test_password_empty_alphabet_raises(self):
        import pytest
        with pytest.raises(ValueError):
            SecureRandom.password(
                use_digits=False,
                use_uppercase=False,
                use_lowercase=False,
                use_punctuation=False,
            )


class TestSecureRandomCompareDigest:
    def test_equal_strings(self):
        assert SecureRandom.compare_digest("abc", "abc") is True

    def test_different_strings(self):
        assert SecureRandom.compare_digest("abc", "xyz") is False

    def test_equal_bytes(self):
        assert SecureRandom.compare_digest(b"abc", b"abc") is True


class TestConvenienceFunctions:
    def test_generate_session_id(self):
        sid = generate_session_id()
        assert len(sid) == 64

    def test_generate_api_key(self):
        key = generate_api_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_generate_csrf_token(self):
        token = generate_csrf_token()
        assert len(token) == 32  # 16 bytes = 32 hex

    def test_generate_oauth_state(self):
        state = generate_oauth_state()
        assert isinstance(state, str)
        assert len(state) > 0
