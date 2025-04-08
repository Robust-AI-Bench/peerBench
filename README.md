 # start of file
# Key Management Module

This module provides cryptographic key management functionality for various blockchain systems.

## Features

- Support for multiple cryptographic types (sr25519, ed25519, ecdsa)
- Key generation, storage, and retrieval
- Signing and verification of messages
- JWT token generation and verification
- Mnemonic phrase support
- Secure key storage

## Usage

```python
from val.key import Key

# Create a new key
key = Key()

# Sign a message
signature = key.sign("Hello, world!")

# Verify a signature
is_valid = key.verify("Hello, world!", signature, key.public_key)

# Generate a JWT token
token = key.get_token({"user_id": 123})

# Verify a JWT token
token_data = key.verify_token(token)
```

## Utils

The module includes a utility library for common cryptographic operations:

- BIP39 mnemonic handling
- ECDSA key operations
- String/bytes conversion utilities
- File system operations for key storage
