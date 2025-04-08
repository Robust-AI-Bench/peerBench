 # start of file
#!/usr/bin/env python3
"""
Key Management Demo Script

This script demonstrates the basic functionality of the Key management module.
"""

from val.key import Key
import json

def main():
    # Create a new key with default settings (ecdsa)
    print("Creating a new ECDSA key...")
    ecdsa_key = Key(crypto_type='ecdsa')
    print(f"Key address: {ecdsa_key.key_address}")
    
    # Create an sr25519 key
    print("\nCreating a new SR25519 key...")
    sr_key = Key(crypto_type='sr25519')
    print(f"Key address: {sr_key.ss58_address}")
    
    # Sign and verify a message
    message = "Hello, blockchain world!"
    print(f"\nSigning message: '{message}'")
    
    # ECDSA signature
    ecdsa_sig = ecdsa_key.sign(message)
    print(f"ECDSA signature: 0x{ecdsa_sig.hex()[:20]}...")
    
    # SR25519 signature
    sr_sig = sr_key.sign(message)
    print(f"SR25519 signature: 0x{sr_sig.hex()[:20]}...")
    
    # Verify signatures
    ecdsa_verify = ecdsa_key.verify(message, ecdsa_sig)
    sr_verify = sr_key.verify(message, sr_sig)
    
    print(f"\nECDSA verification: {ecdsa_verify}")
    print(f"SR25519 verification: {sr_verify}")
    
    # Generate a JWT token
    print("\nGenerating JWT token...")
    token_data = {"user": "demo", "permissions": ["read", "write"]}
    token = ecdsa_key.get_token(token_data)
    
    # Verify the token
    print("Verifying JWT token...")
    verified_data = ecdsa_key.verify_token(token)
    print(f"Token verified with data: {json.dumps(verified_data, indent=2)}")
    
    # Save and load a key
    print("\nSaving key to disk...")
    key_path = "demo_key"
    ecdsa_key.save_json(key_path)
    
    print(f"Loading key from disk: {key_path}")
    loaded_key = Key().get_key(key_path)
    print(f"Loaded key address: {loaded_key.key_address}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
