import hashlib
from typing import *
import os
import threading
import time
import json
from typing import Union, Optional
import time
import os
import re
import base64
from base64 import b64encode
import hashlib
from scalecodec.utils.ss58 import ss58_encode, ss58_decode, get_ss58_format, is_valid_ss58_address
from scalecodec.base import ScaleBytes
from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
import ed25519_zebra
import re
from hashlib import blake2b
import base64
import json
from os import urandom
from typing import Union
from nacl.hashlib import scrypt
from nacl.secret import SecretBox
from sr25519 import pair_from_ed25519_secret_key
from scalecodec.types import Bytes
import hashlib
import hmac
import struct
from eth_keys.datatypes import Signature, PrivateKey
from eth_utils import to_checksum_address, keccak as eth_utils_keccak
from ecdsa.curves import SECP256k1
from random import shuffle

BIP39_PBKDF2_ROUNDS = 2048
BIP39_SALT_MODIFIER = "mnemonic"
BIP32_PRIVDEV = 0x80000000
BIP32_CURVE = SECP256k1
BIP32_SEED_MODIFIER = b'Bitcoin seed'
ETH_DERIVATION_PATH = "m/44'/60'/0'/0"
JUNCTION_ID_LEN = 32
RE_JUNCTION = r'(\/\/?)([^/]+)'
NONCE_LENGTH = 24
SCRYPT_LENGTH = 32 + (3 * 4)
PKCS8_DIVIDER = bytes([161, 35, 3, 33, 0])
PKCS8_HEADER = bytes([48, 83, 2, 1, 1, 48, 5, 6, 3, 43, 101, 112, 4, 34, 4, 32])
PUB_LENGTH = 32
SALT_LENGTH = 32
SEC_LENGTH = 64
SEED_LENGTH = 32
SCRYPT_N = 1 << 15
SCRYPT_P = 1
SCRYPT_R = 8


# path stuff
storage_path = os.path.expanduser('~/.deval')
lib_path = os.path.dirname(os.path.abspath(__file__))
home_path = os.path.expanduser('~')
pwd = os.getcwd()
dir_prefixes  = [pwd, lib_path, home_path]


def sha256(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()


def path2objectpath(path:str, **kwargs) -> str:
    path = os.path.abspath(path)
    for dir_prefix in dir_prefixes:
        if path.startswith(dir_prefix):
            path =  path[len(dir_prefix) + 1:].replace('/', '.')
            break
    if path.endswith('.py'):
        path = path[:-3]
    return path.replace('__init__.', '.')

def detailed_error(e:Exception) -> str:
    import traceback
    return traceback.format_exc()


def get_text(path) -> str:
    with open(path, 'r') as f:
        return f.read()
def abspath(path) -> str:
    return os.path.abspath(os.path.expanduser(path))


def rm_dir(path) -> str:
    import shutil
    path = abspath(path)
    return shutil.rmtree(path)
def ls(path) -> List[str]:
    path = abspath(path)
    try:
        paths =  os.listdir(path)
    except FileNotFoundError:
        return []
    return [os.path.join(path, p) for p in paths]

def glob(path='./', ext='.py') -> List[str]:
    import glob
    path = abspath(path)
    print(f'{path}/**/*{ext}')
    return glob.glob(f'{path}/**/*{ext}', recursive=True)

def path2relative(path) -> str:
    for dir_prefix in dir_prefixes:
        if path.startswith(dir_prefix):
            path =  path[len(dir_prefix) + 1:]
            break
    return path

def path2classes(path='./',
                    class_prefix = 'class ', 
                    file_extension = '.py',
                    tolist = False,
                    depth=4,
                    relative=False,
                    class_suffix = ':', **kwargs) :
    path = abspath(path)
    results = {}
    if os.path.isdir(path) and depth > 0:
        for p in ls(path):
            try:
                for k,v in path2classes(p, depth=depth-1).items():
                    if len(v) > 0:
                        results[k] = v
            except Exception as e:
                print(e)
                pass
    elif os.path.isfile(path) and path.endswith('.py'):
        code = get_text(path)
        classes = []
        file_path = path2objectpath(path)
        for line in code.split('\n'):
            if line.startswith(class_prefix) and line.strip().endswith(class_suffix):
                new_class = line.split(class_prefix)[-1].split('(')[0].strip()
                if new_class.endswith(class_suffix):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
        if file_path.startswith(path):
            file_path = file_path[len(path)+1:]
        if '/' in file_path:
            file_path = file_path.replace('/', '.')
        if relative:
            path = path2relative(path)
        results =  {path:  [file_path + '.' + cl for cl in classes]}
    if tolist: 
        classes = []
        for k,v in results.items():
            classes.extend(v)
        return classes

    return results


def shorten_name(name:str, max_length=20) -> str:
    chunks = name.split('.')[1:-1]
    new_name = ''
    for chunk in chunks:
        if len(new_name) + len(chunk) > max_length:
            break
        if chunk in new_name:
            continue
        new_name += chunk + '.'
    if new_name.endswith('.'):
        new_name = new_name[:-1]
    return new_name



def objs(path='./', **kwargs) -> List[str]:
    p2c =  path2classes(path, **kwargs)
    classes = []
    for k,v in p2c.items():
        classes.extend(v)
    
    return classes

def tree(path='./', search=None,  **kwargs) -> Dict[str, List[str]]:
    result = {}
    for o in objs(path, **kwargs):
        short_name = shorten_name(o)
        name = o
        if search and search not in name:
            continue
        result[short_name] = name
    
    return result


def modules(path='./', search=None, **kwargs) -> List[str]:
    return list(tree(path=path, search=search).keys())

def obj(name:str, **kwargs):
    if '.' in name:
        module_name = '.'.join(name.split('.')[:-1])
        class_name = name.split('.')[-1]
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        return __import__(name)

def module(name:str, **kwargs):
    _tree = tree()
    if name in _tree:
        name = _tree[name]
    else:
        for k,v in _tree.items():
            if name in v:
                name = v
                break
    return obj(name)




def submit(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    t.start()
    return t



def as_completed(futures, timeout=10):
    start = time.time()
    for f in futures:
        f.join(timeout)
        if time.time() - start > timeout:
            break
        if f.is_alive():
            continue
        yield f


def thread(fn: Union['callable', str],  
                args:list = None, 
                kwargs:dict = None, 
                daemon:bool = True, 
                name = None,
                tag = None,
                start:bool = True,
                tag_seperator:str='::', 
                **extra_kwargs):
    import threading
    import commune as c
    
    if args == None:
        args = []
    if kwargs == None:
        kwargs = {}

    assert callable(fn), f'target must be callable, got {fn}'
    assert  isinstance(args, list), f'args must be a list, got {args}'
    assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
    
    # unique thread name
    if name == None:
        name = fn.__name__
        cnt = 0
        while name in thread_map:
            cnt += 1
            if tag == None:
                tag = ''
            name = name + tag_seperator + tag + str(cnt)
    
    if name in thread_map:
        thread_map[name].join()

    t = threading.Thread(target=fn, args=args, kwargs=kwargs, **extra_kwargs)
    # set the time it starts
    t.daemon = daemon
    if start:
        t.start()
    thread_map[name] = t
    return t



def wait(futures, timeout=10):
    start = time.time()
    for f in futures:
        f.join(timeout)
        if time.time() - start > timeout:
            break
    return futures

def code(x):
    import inspect
    return inspect.getsource(x)



def is_valid_ss58_address(address: str) -> bool:
    """
    Check if the given address is a valid SS58 address
    """
    try:
        ss58_decode(address)
        return True
    except:
        return False


def str2bytes( data: str, mode: str = 'hex') -> bytes:
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        return bytes.fromhex(data)



def is_valid_ecdsa_address(address: str) -> bool:
    """
    Check if the given address is a valid ECDSA address
    """
    try:
        return len(bytes.fromhex(address)) == 20
    except:
        return False
class PublicKey:
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
        return to_checksum_address(eth_utils_keccak(s)[12:])

def mnemonic_to_bip39seed(mnemonic, passphrase):
    mnemonic = bytes(mnemonic, 'utf8')
    salt = bytes(BIP39_SALT_MODIFIER + passphrase, 'utf8')
    return hashlib.pbkdf2_hmac('sha512', mnemonic, salt, BIP39_PBKDF2_ROUNDS)

def bip39seed_to_bip32masternode(seed):
    h = hmac.new(BIP32_SEED_MODIFIER, seed, hashlib.sha512).digest()
    key, chain_code = h[:32], h[32:]
    return key, chain_code

def derive_bip32childkey(parent_key, parent_chain_code, i):
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
    signer = PrivateKey(private_key)
    return signer.sign_msg(message).to_bytes()


def ecdsa_verify(signature: bytes, data: bytes, address: bytes) -> bool:
    signature_obj = Signature(signature)
    recovered_pubkey = signature_obj.recover_public_key_from_msg(data)
    return recovered_pubkey.to_canonical_address() == address


def decode_pair_from_encrypted_json(json_data: Union[str, dict], passphrase: str) -> tuple:
    """
    Decodes encrypted PKCS#8 message from PolkadotJS JSON format

    Parameters
    ----------
    json_data
    passphrase

    Returns
    -------
    tuple containing private and public key
    """
    if type(json_data) is str:
        json_data = json.loads(json_data)

    # Check requirements
    if json_data.get('encoding', {}).get('version') != "3":
        raise ValueError("Unsupported JSON format")

    encrypted = base64.b64decode(json_data['encoded'])

    if 'scrypt' in json_data['encoding']['type']:
        salt = encrypted[0:32]
        n = int.from_bytes(encrypted[32:36], byteorder='little')
        p = int.from_bytes(encrypted[36:40], byteorder='little')
        r = int.from_bytes(encrypted[40:44], byteorder='little')

        password = scrypt(passphrase.encode(), salt, n=n, r=r, p=p, dklen=32, maxmem=2 ** 26)
        encrypted = encrypted[SCRYPT_LENGTH:]

    else:
        password = passphrase.encode().rjust(32, b'\x00')

    if "xsalsa20-poly1305" not in json_data['encoding']['type']:
        raise ValueError("Unsupported encoding type")

    nonce = encrypted[0:NONCE_LENGTH]
    message = encrypted[NONCE_LENGTH:]

    secret_box = SecretBox(key=password)
    decrypted = secret_box.decrypt(message, nonce)

    # Decode PKCS8 message
    secret_key, public_key = decode_pkcs8(decrypted)

    if 'sr25519' in json_data['encoding']['content']:
        # Secret key from PolkadotJS is an Ed25519 expanded secret key, so has to be converted
        # https://github.com/polkadot-js/wasm/blob/master/packages/wasm-crypto/src/rs/sr25519.rs#L125
        converted_public_key, secret_key = pair_from_ed25519_secret_key(secret_key)
        assert(public_key == converted_public_key)

    return secret_key, public_key

def decode_pkcs8(ciphertext: bytes) -> tuple:
    current_offset = 0

    header = ciphertext[current_offset:len(PKCS8_HEADER)]
    if header != PKCS8_HEADER:
        raise ValueError("Invalid Pkcs8 header found in body")

    current_offset += len(PKCS8_HEADER)

    secret_key = ciphertext[current_offset:current_offset + SEC_LENGTH]
    current_offset += SEC_LENGTH

    divider = ciphertext[current_offset:current_offset + len(PKCS8_DIVIDER)]

    if divider != PKCS8_DIVIDER:
        raise ValueError("Invalid Pkcs8 divider found in body")

    current_offset += len(PKCS8_DIVIDER)

    public_key = ciphertext[current_offset: current_offset + PUB_LENGTH]

    return secret_key, public_key

def encode_pkcs8(public_key: bytes, private_key: bytes) -> bytes:
    return PKCS8_HEADER + private_key + PKCS8_DIVIDER + public_key

def encode_pair(public_key: bytes, private_key: bytes, passphrase: str) -> bytes:
    """
    Encode a public/private pair to PKCS#8 format, encrypted with provided passphrase

    Parameters
    ----------
    public_key: 32 bytes public key
    private_key: 64 bytes private key
    passphrase: passphrase to encrypt the PKCS#8 message

    Returns
    -------
    (Encrypted) PKCS#8 message bytes
    """
    message = encode_pkcs8(public_key, private_key)

    salt = urandom(SALT_LENGTH)
    password = scrypt(passphrase.encode(), salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32, maxmem=2 ** 26)

    secret_box = SecretBox(key=password)
    message = secret_box.encrypt(message)

    scrypt_params = SCRYPT_N.to_bytes(4, 'little') + SCRYPT_P.to_bytes(4, 'little') + SCRYPT_R.to_bytes(4, 'little')

    return salt + scrypt_params + message.nonce + message.ciphertext




def python2str(x):
    from copy import deepcopy
    import json
    x = deepcopy(x)
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
        
    return x

class DeriveJunction:
    def __init__(self, chain_code, is_hard=False):
        self.chain_code = chain_code
        self.is_hard = is_hard

    @classmethod
    def from_derive_path(cls, path: str, is_hard=False):

        if path.isnumeric():
            byte_length = ceil(int(path).bit_length() / 8)
            chain_code = int(path).to_bytes(byte_length, 'little').ljust(32, b'\x00')

        else:
            path_scale = Bytes()
            path_scale.encode(path)

            if len(path_scale.data) > JUNCTION_ID_LEN:
                chain_code = blake2b(path_scale.data.data, digest_size=32).digest()
            else:
                chain_code = bytes(path_scale.data.data.ljust(32, b'\x00'))

        return cls(chain_code=chain_code, is_hard=is_hard)

def extract_derive_path(derive_path: str):

    path_check = ''
    junctions = []
    paths = re.findall(RE_JUNCTION, derive_path)

    if paths:
        path_check = ''.join(''.join(path) for path in paths)

        for path_separator, path_value in paths:
            junctions.append(DeriveJunction.from_derive_path(
                path=path_value, is_hard=path_separator == '//')
            )

    if path_check != derive_path:
        raise ValueError('Reconstructed path "{}" does not match input'.format(path_check))

    return junctions

def bytes2str(data: bytes, mode: str = 'utf-8') -> str:
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)



def valid_h160_address(cls, address):
    # Check if it starts with '0x'
    if not address.startswith('0x'):
        return False
    
    # Remove '0x' prefix
    address = address[2:]
    
    # Check length
    if len(address) != 40:
        return False
    
    # Check if it contains only valid hex characters
    if not re.match('^[0-9a-fA-F]{40}$', address):
        return False
    
    return True

def is_mnemonic(mnemonic:str, language_code='en') -> bool:
    """
    Check if the provided string is a valid mnemonic
    """
    if not isinstance(mnemonic, str):
        return False
    return bip39_validate(mnemonic, language_code)

def get_json( path):
    with open(path) as f:
        return json.load(f)

def put_json( path, data):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


def is_int(x):
    try:
        int(x)
        return True
    except:
        return False

def abspath(path):
    return os.path.abspath(os.path.expanduser(path))

def import_module(import_path:str ) -> 'Object':
    from importlib import import_module
    return import_module(import_path)

def df(x):
    import pandas as pd
    return pd.DataFrame(x)


def str2python( x):
    x = str(x)
    if isinstance(x, str) :
        if x.startswith('py(') and x.endswith(')'):
            try:
                return eval(x[3:-1])
            except:
                return x
    if x.lower() in ['null'] or x == 'None':  # convert 'null' or 'None' to None
        return None 
    elif x.lower() in ['true', 'false']: # convert 'true' or 'false' to bool
        return bool(x.lower() == 'true')
    elif x.startswith('[') and x.endswith(']'): # this is a list
        try:
            list_items = x[1:-1].split(',')
            # try to convert each item to its actual type
            x =  [str2python(item.strip()) for item in list_items]
            if len(x) == 1 and x[0] == '':
                x = []
            return x
        except:
            # if conversion fails, return as string
            return x
    elif x.startswith('{') and x.endswith('}'):
        # this is a dictionary
        if len(x) == 2:
            return {}
        try:
            dict_items = x[1:-1].split(',')
            # try to convert each item to a key-value pair
            return {key.strip(): str2python(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
        except:
            # if conversion fails, return as string
            return x
    else:
        # try to convert to int or float, otherwise return as string
        
        for type_fn in [int, float]:
            try:
                return type_fn(x)
            except ValueError:
                pass
    return x

def _base64url_encode(self, data):
    """Encode data in base64url format"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, dict):
        data = json.dumps(data, separators=(',', ':')).encode('utf-8')
    encoded = base64.urlsafe_b64encode(data).rstrip(b'=')
    return encoded.decode('utf-8')

def _base64url_decode(self, data):
    """Decode base64url data"""
    padding = b'=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data.encode('utf-8') + padding)


def is_generator(x):
    return hasattr(x, '__next__') or hasattr(x, '__iter__')