 # start of file
import hashlib
from typing import *
import os
import time
import json
import hmac
import re
import base64
import base58
import struct
from eth_keys.datatypes import Signature, PrivateKey
from eth_utils import to_checksum_address, keccak as eth_utils_keccak
from ecdsa.curves import SECP256k1
from scalecodec.utils.ss58 import ss58_encode, ss58_decode, get_ss58_format, is_valid_ss58_address

# Constants for cryptographic operations
BIP39_PBKDF2_ROUNDS = 2048
BIP39_SALT_MODIFIER = "mnemonic"
BIP32_PRIVDEV = 0x80000000
BIP32_CURVE = SECP256k1
BIP32_SEED_MODIFIER = b'Bitcoin seed'
ETH_DERIVATION_PATH = "m/44'/60'/0'/0"

def is_int(x):
    """Check if a value can be converted to an integer"""
    try:
        int(x)
        return True
    except:
        return False

def str2bytes(data: str, mode: str = 'hex') -> bytes:
    """Convert string to bytes using the specified encoding mode"""
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        if data.startswith('0x'):
            data = data[2:]
        return bytes.fromhex(data)

def python2str(x):
    """Convert Python object to string representation"""
    import copy
    import json
    x = copy.deepcopy(x)
    input_type = type(x)
    if input_type == str:
        return x
    if input_type in [dict]:
        x = json.dumps(x)
    elif input_type in [bytes]:
        x = bytes2str(x)
    elif input_type in [list, tuple, set]:
        x = json.dumps(list(x))
    elif input_type in [int, float, bool]:
        x = str(x)
    # remove the quotes
    if isinstance(x, str) and len(x) > 2 and x[0] == '"' and x[-1] == '"':
        x = x[1:-1]
    return x

def bytes2str(data: bytes, mode: str = 'utf-8') -> str:
    """Convert bytes to string using the specified encoding"""
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)

def abspath(path):
    """Get absolute path from a possibly relative path"""
    return os.path.abspath(os.path.expanduser(path))

def rm_dir(path) -> str:
    """Remove directory and all its contents"""
    import shutil
    path = abspath(path)
    return shutil.rmtree(path)

def ls(path) -> List[str]:
    """List contents of a directory"""
    path = abspath(path)
    try:
        paths = os.listdir(path)
    except FileNotFoundError:
        return []
    return [os.path.join(path, p) for p in paths]

def get_json(path):
    """Load JSON from a file"""
    with open(path) as f:
        return json.load(f)

def put_json(path, data):
    """Save data as JSON to a file"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    return path

class PublicKey:
    """Helper class for ECDSA public key operations"""
    def __init__(self, private_key):
        self.point = int.from_bytes(private_key, byteorder='big') * BIP32_CURVE.generator

    def __bytes__(self):
        xstr = int(self.point.x()).to_bytes(32, byteorder='big')
        parity = int(self.point.y()) & 1
        return (2 + parity).to_bytes(1, byteorder='big') + xstr

    def address(self):
        x = int(self.point.x())
        y = int(self.point.y())
        s = x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        return to_checksum_address(keccak(s)[12:])

def mnemonic_to_bip39seed(mnemonic, passphrase):
    """Convert mnemonic to BIP39 seed"""
    mnemonic = bytes(mnemonic, 'utf8')
    salt = bytes(BIP39_SALT_MODIFIER + passphrase, 'utf8')
    return hashlib.pbkdf2_hmac('sha512', mnemonic, salt, BIP39_PBKDF2_ROUNDS)

def bip39seed_to_bip32masternode(seed):
    """Convert BIP39 seed to BIP32 master node"""
    h = hmac.new(BIP32_SEED_MODIFIER, seed, hashlib.sha512).digest()
    key, chain_code = h[:32], h[32:]
    return key, chain_code

def derive_bip32childkey(parent_key, parent_chain_code, i):
    """Derive BIP32 child key"""
    assert len(parent_key) == 32
    assert len(parent_chain_code) == 32
    k = parent_chain_code
    if (i & BIP32_PRIVDEV) != 0:
        key = b'\x00' + parent_key
    else:
        key = bytes(PublicKey(parent_key))
    d = key + struct.pack('>L', i)
    while True:
        h = hmac.new(k, d, hashlib.sha512).digest()
        key, chain_code = h[:32], h[32:]
        a = int.from_bytes(key, byteorder='big')
        b = int.from_bytes(parent_key, byteorder='big')
        key = (a + b) % int(BIP32_CURVE.order)
        if a < BIP32_CURVE.order and key != 0:
            key = key.to_bytes(32, byteorder='big')
            break
        d = b'\x01' + h[32:] + struct.pack('>L', i)
    return key, chain_code

def parse_derivation_path(str_derivation_path):
    """Parse derivation path string to path components"""
    path = []
    if str_derivation_path[0:2] != 'm/':
        raise ValueError("Can't recognize derivation path. It should look like \"m/44'/60/0'/0\".")
    for i in str_derivation_path.lstrip('m/').split('/'):
        if "'" in i:
            path.append(BIP32_PRIVDEV + int(i[:-1]))
        else:
            path.append(int(i))
    return path

def mnemonic_to_ecdsa_private_key(mnemonic: str, str_derivation_path: str = None, passphrase: str = "") -> bytes:
    """Convert mnemonic to ECDSA private key"""
    import hmac
    
    if str_derivation_path is None:
        str_derivation_path = f'{ETH_DERIVATION_PATH}/0'

    derivation_path = parse_derivation_path(str_derivation_path)
    bip39seed = mnemonic_to_bip39seed(mnemonic, passphrase)
    master_private_key, master_chain_code = bip39seed_to_bip32masternode(bip39seed)
    private_key, chain_code = master_private_key, master_chain_code
    for i in derivation_path:
        private_key, chain_code = derive_bip32childkey(private_key, chain_code, i)
    return private_key

def ecdsa_sign(private_key: bytes, message: bytes) -> bytes:
    """Sign a message using ECDSA"""
    signer = PrivateKey(private_key)
    return signer.sign_msg(message).to_bytes()

def ecdsa_verify(signature: bytes, data: bytes, address: bytes) -> bool:
    """Verify an ECDSA signature"""
    signature_obj = Signature(signature)
    recovered_pubkey = signature_obj.recover_public_key_from_msg(data)
    return recovered_pubkey.to_canonical_address() == address
