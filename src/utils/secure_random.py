"""
Secure Random Number Generation Utilities
==========================================
Provides cryptographically secure random number generation for security-sensitive contexts.

CRITICAL: This module MUST be used for:
- Authentication tokens (session IDs, API keys, access tokens)
- Cryptographic keys and salts
- Password reset tokens
- CSRF tokens
- OAuth state parameters and nonces
- Any value where unpredictability is a security requirement

DO NOT use this module for:
- Machine learning (epsilon-greedy exploration, experience sampling)
- Simulations and modeling
- Test data generation (where reproducibility is needed)
- Performance-critical non-security code

For non-security randomness, use:
- random module: for simple randomness
- numpy.random: for numerical/ML applications with seeded reproducibility
"""

import secrets
import string


class SecureRandom:
    """
    Cryptographically secure random number generation.

    Uses Python's secrets module, which is designed for security-sensitive applications.
    All methods use a CSPRNG (cryptographically secure pseudorandom number generator)
    that is suitable for managing data such as passwords, account authentication,
    security tokens, and related secrets.

    Reference: https://docs.python.org/3/library/secrets.html
    """

    @staticmethod
    def token_hex(nbytes: int = 32) -> str:
        """
        Generate a secure random hex token.

        Args:
            nbytes: Number of random bytes (default 32)

        Returns:
            Hex string of length 2*nbytes (each byte = 2 hex chars)

        Example:
            >>> token = SecureRandom.token_hex(16)
            >>> len(token)
            32
        """
        return secrets.token_hex(nbytes)

    @staticmethod
    def token_urlsafe(nbytes: int = 32) -> str:
        """
        Generate a secure random URL-safe token.

        Uses base64 encoding, safe for URLs and filenames.

        Args:
            nbytes: Number of random bytes (default 32)

        Returns:
            URL-safe base64 string

        Example:
            >>> api_key = SecureRandom.token_urlsafe(32)
        """
        return secrets.token_urlsafe(nbytes)

    @staticmethod
    def token_bytes(nbytes: int = 32) -> bytes:
        """
        Generate secure random bytes.

        Args:
            nbytes: Number of random bytes (default 32)

        Returns:
            Random bytes

        Example:
            >>> key = SecureRandom.token_bytes(32)  # 256-bit key
        """
        return secrets.token_bytes(nbytes)

    @staticmethod
    def randbelow(exclusive_upper_bound: int) -> int:
        """
        Generate a secure random integer in range [0, n).

        Args:
            exclusive_upper_bound: Upper bound (exclusive)

        Returns:
            Random integer k such that 0 <= k < exclusive_upper_bound

        Example:
            >>> dice_roll = SecureRandom.randbelow(6) + 1  # 1-6
        """
        return secrets.randbelow(exclusive_upper_bound)

    @staticmethod
    def choice(sequence: list | tuple | str):
        """
        Choose a secure random element from a sequence.

        Args:
            sequence: Non-empty sequence

        Returns:
            Random element from sequence

        Example:
            >>> action = SecureRandom.choice(['allow', 'deny'])
        """
        return secrets.choice(sequence)

    @staticmethod
    def session_token(length: int = 32) -> str:
        """
        Generate a secure session token.

        Suitable for session IDs, CSRF tokens, etc.

        Args:
            length: Token length in bytes (default 32 = 256 bits)

        Returns:
            Hex token suitable for session management

        Example:
            >>> session_id = SecureRandom.session_token()
        """
        return secrets.token_hex(length)

    @staticmethod
    def api_key(length: int = 32) -> str:
        """
        Generate a secure API key.

        URL-safe format, suitable for API authentication.

        Args:
            length: Key length in bytes (default 32 = 256 bits)

        Returns:
            URL-safe token suitable for API keys

        Example:
            >>> api_key = SecureRandom.api_key()
        """
        return secrets.token_urlsafe(length)

    @staticmethod
    def password(
        length: int = 16,
        use_digits: bool = True,
        use_uppercase: bool = True,
        use_lowercase: bool = True,
        use_punctuation: bool = False,
    ) -> str:
        """
        Generate a secure random password.

        Args:
            length: Password length (default 16)
            use_digits: Include digits 0-9
            use_uppercase: Include uppercase A-Z
            use_lowercase: Include lowercase a-z
            use_punctuation: Include punctuation symbols

        Returns:
            Secure random password

        Example:
            >>> pwd = SecureRandom.password(20, use_punctuation=True)
        """
        alphabet = ""
        if use_lowercase:
            alphabet += string.ascii_lowercase
        if use_uppercase:
            alphabet += string.ascii_uppercase
        if use_digits:
            alphabet += string.digits
        if use_punctuation:
            alphabet += string.punctuation

        if not alphabet:
            raise ValueError("At least one character set must be enabled")

        # Generate password with at least one character from each enabled set
        password_chars = []

        # Ensure at least one char from each enabled set
        if use_lowercase:
            password_chars.append(secrets.choice(string.ascii_lowercase))
        if use_uppercase:
            password_chars.append(secrets.choice(string.ascii_uppercase))
        if use_digits:
            password_chars.append(secrets.choice(string.digits))
        if use_punctuation:
            password_chars.append(secrets.choice(string.punctuation))

        # Fill remaining length
        for _ in range(length - len(password_chars)):
            password_chars.append(secrets.choice(alphabet))

        # Shuffle to avoid predictable pattern
        # Using Fisher-Yates shuffle with secrets
        for i in range(len(password_chars) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_chars[i], password_chars[j] = password_chars[j], password_chars[i]

        return "".join(password_chars)

    @staticmethod
    def nonce(nbytes: int = 16) -> str:
        """
        Generate a cryptographic nonce (number used once).

        Suitable for OAuth state parameters, challenge-response, etc.

        Args:
            nbytes: Nonce length in bytes (default 16 = 128 bits)

        Returns:
            Hex nonce

        Example:
            >>> nonce = SecureRandom.nonce()
        """
        return secrets.token_hex(nbytes)

    @staticmethod
    def compare_digest(a: str | bytes, b: str | bytes) -> bool:
        """
        Timing-attack resistant string comparison.

        Use this to compare security tokens, passwords, etc.
        Regular == comparison can leak timing information.

        Args:
            a: First value
            b: Second value

        Returns:
            True if equal, False otherwise

        Example:
            >>> if SecureRandom.compare_digest(provided_token, stored_token):
            ...     # Token is valid
        """
        return secrets.compare_digest(a, b)


# Convenience functions for common use cases


def generate_session_id() -> str:
    """Generate a secure session ID (32 bytes = 64 hex chars)."""
    return SecureRandom.session_token(32)


def generate_api_key() -> str:
    """Generate a secure API key (32 bytes, URL-safe)."""
    return SecureRandom.api_key(32)


def generate_csrf_token() -> str:
    """Generate a CSRF token (16 bytes = 32 hex chars)."""
    return SecureRandom.session_token(16)


def generate_oauth_state() -> str:
    """Generate OAuth state parameter (32 bytes, URL-safe)."""
    return SecureRandom.api_key(32)


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("Secure Random Number Generation Examples")
    print("=" * 70)

    print("\n[1] Session Token (hex):")
    print(f"    {SecureRandom.session_token()}")

    print("\n[2] API Key (URL-safe):")
    print(f"    {SecureRandom.api_key()}")

    print("\n[3] Random Password:")
    print(f"    {SecureRandom.password(16, use_punctuation=True)}")

    print("\n[4] OAuth Nonce:")
    print(f"    {SecureRandom.nonce()}")

    print("\n[5] Random Integer (0-99):")
    print(f"    {SecureRandom.randbelow(100)}")

    print("\n[6] Random Choice:")
    choices = ["approve", "deny", "pending"]
    print(f"    {SecureRandom.choice(choices)}")

    print("\n[7] CSRF Token:")
    print(f"    {generate_csrf_token()}")

    print("\n[8] Timing-safe comparison:")
    token1 = SecureRandom.session_token()
    token2 = SecureRandom.session_token()
    print(f"    Different tokens equal? {SecureRandom.compare_digest(token1, token2)}")
    print(f"    Same token equal? {SecureRandom.compare_digest(token1, token1)}")

    print("\n" + "=" * 70)
    print("✓ All examples completed")
    print("=" * 70)
