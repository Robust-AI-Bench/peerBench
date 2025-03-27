
from typing import Union, Optional
import time
import os
import binascii
import re
import secrets
import base64
import hashlib
import nacl.bindings
import nacl.public
from scalecodec.base import ScaleBytes
from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
from sr25519 import pair_from_ed25519_secret_key
import ed25519_zebra
import re
from hashlib import blake2b
import json
from scalecodec.types import Bytes
import hashlib
from copy import deepcopy
import hmac
import struct
from eth_keys.datatypes import Signature, PrivateKey
from .utils import *

# imoport 

class Key:
    crypto_type_map = {'ed25519': 0, 'sr25519': 1, 'ecdsa': 2}

    def __init__(self,
                 private_key: Union[bytes, str] = None, 
                 mnemonic : Optional[str] = None,
                crypto_type =  'ecdsa',
                 path : Optional[str] = None,
                ss58_format = 42,
                language_code = 'en',
                storage_path = '~/.deval/key',
                 **kwargs): 
        self.ss58_format = ss58_format
        self.crypto_type = crypto_type
        self.language_code = language_code
        self.reverse_crypto_type_map = {v:k for k,v in self.crypto_type_map.items()}
        self.storage_path = os.path.expanduser(storage_path)
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.set_key(private_key=private_key, crypto_type=crypto_type, mnemonic=mnemonic, path=path, **kwargs)

    def set_key(self, private_key: Union[bytes, str] ,  crypto_type: int , mnemonic:Optional[str] = None, path=None, **kwargs):
        """
        Allows generation of Keys from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these Keys data can be signed and verified

        Parameters
        ----------
        private_key: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        seed_hex: hex string of seed
        crypto_type: Use "sr25519" or "ed25519"cryptography for generating the Key
        """
        if path != None:
            private_key = self.get_key(path, crypto_type=crypto_type).private_key
        self.crypto_type = crypto_type = self.get_crypto_type(crypto_type)
        if  mnemonic:
            private_key = self.from_mnemonic(mnemonic, crypto_type=crypto_type).private_key
        if private_key == None:
            private_key = self.new_key(crypto_type=crypto_type).private_key
        if type(private_key) == str:
            private_key = str2bytes(private_key)
        # If the private key is longer than 32 bytes, it is assumed to be a seed and the public key is derived from it
        if crypto_type == 'sr25519':
            if len(private_key) != 64:
                private_key = sr25519.pair_from_seed(private_key)[1]
            public_key = sr25519.public_from_secret_key(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        elif crypto_type == "ecdsa":
            private_key = private_key[0:32] if len(private_key) > 32 else private_key
            assert len(private_key) == 32, f'private_key should be 32 bytes, got {len(private_key)}'
            private_key_obj = PrivateKey(private_key)
            public_key = private_key_obj.public_key.to_address()
            key_address = private_key_obj.public_key.to_checksum_address()
        elif crypto_type == "ed25519":       
            private_key = private_key[:32]    
            assert len(private_key) == 32  
            public_key, private_key = ed25519_zebra.ed_from_seed(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))

        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))
        self.private_key = private_key
        self.public_key = public_key
        self.key_address = self.address = self.ss58_address =  key_address
        self.multiaddress = f'{crypto_type}/{key_address}'
        return {'key_address':key_address, 'crypto_type':crypto_type, 'multiaddress':self.multiaddress}

    def get_crypto_type(self, crypto_type: Union[int, str]=None) -> str:
        """
        returns the crypto type as a string (e.g. 'sr25519') from the crypto type integer (e.g. 1)
        """
        if  crypto_type == None:
            crypto_type =  self.crypto_type
        if is_int(crypto_type):
            crypto_type = self.reverse_crypto_type_map[int(crypto_type)]
        elif isinstance(crypto_type, str):
            crypto_type = crypto_type.lower()
        else: 
            raise ValueError(f'crypto_type {crypto_type} not supported')
        return crypto_type

    @property
    def shorty(self):
        n = 4
        return self.key_address[:n] + '...' + self.key_address[-n:]

    def add_key(self, path:str,  crypto_type=None, mnemonic:str = None, refresh:bool=True, private_key=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key_exists = self.key_exists(path, crypto_type=crypto_type)
        if not key_exists or refresh :
            key = self.new_key( private_key=private_key, crypto_type=crypto_type, mnemonic=mnemonic, **kwargs)
            key.save_json(path)
            print(f'key added at {path}')
            assert self.key_exists(path, crypto_type=crypto_type), f'key does not exist at {path}'
        return self.get_key(path, crypto_type=crypto_type)

    def save_json(self, path:str):
        crypto_type = self.get_crypto_type(self.crypto_type)
        key_json = json.loads(self.to_json())
        key_json_path = self.resolve_path(path) + '/' + crypto_type+ '/' + self.key_address + '.json'
        put_json(key_json_path, key_json)
        print(f'key saved at {key_json_path}')
        return key_json_path
            
    def resolve_path(self, path:str) -> str:
        path = str(path)
        if not path.startswith(self.storage_path):
            path = self.storage_path + '/' + path
        return path

    def get_key(self, 
                key:str,
                create_if_not_exists:bool = True, 
                crypto_type=None, 
                **kwargs):

        crypto_type = self.get_crypto_type(crypto_type)
        if hasattr(key, 'key_address'):
            return key
        if 'type' in kwargs:
            crypto_type = kwargs.pop('type')
        key = key or 'module'
        if not self.key_exists(key):
            if create_if_not_exists:
                key_obj = self.add_key(key, crypto_type=crypto_type,  **kwargs) # create key
            else:
                raise ValueError(f'key does not exist at --> {path}')
        return self.from_path(key, crypto_type=crypto_type, **kwargs)

    def from_path(self, path:str, crypto_type=None, **kwargs):
        key_json = self.get_key_data(path, crypto_type=crypto_type)
        key_json = json.loads(key_json) if isinstance(key_json, str) else key_json
        key =  self.from_json(key_json, crypto_type=crypto_type)
        return key

    def key2path(self, crypto_type=None) -> dict:
        """
        defines the path for each key
        """
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = {}
        for p in  ls(self.storage_path):
            print(f'checking path {p}')
            files = ls(f'{p}/{crypto_type}')
            if len(files) >= 1:
                file2age = {f:os.path.getmtime(f) for f in files}
                files = [k for k,v in sorted(file2age.items(), key=lambda item: item[1])]
                # get the latest file
                p = files[0]
                # delete the others
                for f in files[1:]:
                    os.remove(f)
                name = p.split('/')[-3]
                key2path[name] = p         
        return key2path
    
    def key2address(self, search=None, crypto_type=None,  **kwargs):
        key2path = self.key2path(crypto_type=crypto_type)
        key2address = {}
        for key, path in key2path.items():
            key2address[key] = path.split('/')[-1].split('.')[0]
        return key2address

    def key2type(self, search=None, crypto_type=None,  **kwargs):
        key2path = self.key2path(crypto_type=crypto_type)
        key2address = {}
        for key, path in key2path.items():
            key2address[key] = path.split('/')[-1].split('.')[0]
        return key2address

    def address2key(self, search:Optional[str]=None,  crypto_type=None, **kwargs):
        address2key =  { v: k for k,v in self.key2address(crypto_type=crypto_type).items()}
        if search != None :
            return {k:v for k,v in address2key.items() if search in k}
        return address2key
    
    def keys(self, search : str = None, crypto_type=None, **kwargs):
        keys = list(self.key2path(crypto_type=crypto_type).keys())
        if search != None:
            keys = [key for key in keys if search in key]
        return keys
    
    def key_exists(self, key, crypto_type=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key_path = self.storage_path + '/' + key +'/' + crypto_type + '/' 
        return os.path.exists(key_path)
    
    def get_key_path(self, key, crypto_type=None):
        crypto_type = self.get_crypto_type(crypto_type)
        path = self.storage_path + '/' + key +'/' + crypto_type + '/'
        result = None
        try:
            paths = os.listdir(path)
            if len(paths) > 0:
                result =  path + paths[0]
        except:
            pass
        print(f'getting key path at {path}', paths)
        return result

    def get_key_data(self, key, crypto_type=None):
        key_path =  self.get_key_path(key, crypto_type=crypto_type)
        if key_path == None:
            return None
        output =  get_json(key_path)
        # if single quoted json, convert to double quoted json string and load
        if isinstance(output, str):
            output = output.replace("'", '"')
        return json.loads(output) if isinstance(output, str) else output

    
    def rm_key(self, key=None, crypto_type=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = self.key2path(crypto_type=crypto_type)
        keys = list(key2path.keys())
        if key not in keys:
            return {'msg':'key not found'}
        else:
            path = os.path.dirname(key2path[key])
            
            assert path.endswith(f'{key}/{crypto_type}'), f'invalid path {path}'
            rm_dir(path)
            assert not self.key_exists(key, crypto_type=crypto_type), f'key {key} exists'
            return {'msg':'key deleted', 'key':key, 'path':path, 'crypto_type':crypto_type}

    def new_key(self, mnemonic:str = None, suri:str = None, private_key: str = None, crypto_type: Union[int,str] = None,  **kwargs):
        '''
        yo rody, this is a class method you can gen keys whenever fam
        '''
        crypto_type = self.get_crypto_type(crypto_type)
        if mnemonic:
            key = self.from_mnemonic(mnemonic, crypto_type=crypto_type)
        elif private_key:
            key = self.from_private_key(private_key,crypto_type=crypto_type)
        elif suri:
            key =  self.from_uri(suri, crypto_type=crypto_type)
        else:
            key = self.from_mnemonic(self.generate_mnemonic(), crypto_type=crypto_type)
            
        return key
        
    def to_json(self) -> dict:
        from copy import copy
        state_dict =  deepcopy(self.__dict__)
        for k,v in state_dict.items():
            if type(v)  in [bytes]:
                state_dict[k] = v.hex() 
        if '_ss58_address' in state_dict:
            state_dict['ss58_address'] = state_dict.pop('_ss58_address')
        state_dict = json.dumps(state_dict)
        return state_dict
    
    def from_json(self, obj: Union[str, dict],crypto_type=None) -> dict:
        if type(obj) == str:
            obj = json.loads(obj)
        if obj == None:
           return None 
        if 'ss58_address' in obj:
            obj['_ss58_address'] = obj.pop('ss58_address')
        if crypto_type != None:
            obj['crypto_type'] = crypto_type
        return  Key(**obj)

    def generate_mnemonic(self, words: int = 24) -> str:
        """
        params:
            words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24
        """
        mnemonic =  bip39_generate(words, self.language_code)
        assert bip39_validate(mnemonic, self.language_code), """Invalid mnemonic, please provide a valid mnemonic"""
        return mnemonic
        
    def from_mnemonic(self, mnemonic: str = None, crypto_type=None) -> 'Key':
        """
        Create a Key for given memonic
        """

        crypto_type = self.get_crypto_type(crypto_type)
        mnemonic = mnemonic or self.generate_mnemonic()
        if crypto_type == "ecdsa":
            if self.language_code != "en":
                raise ValueError("ECDSA mnemonic only supports english")
            peivate_key = mnemonic_to_ecdsa_private_key(mnemonic)
            keypair = self.from_private_key(mnemonic_to_ecdsa_private_key(mnemonic), crypto_type=crypto_type)
        else:
            seed_hex = binascii.hexlify(bytearray(bip39_to_mini_secret(mnemonic, "", self.language_code))).decode("ascii")
            if type(seed_hex) is str:
                seed_hex = bytes.fromhex(seed_hex.replace('0x', ''))
            if crypto_type == 'sr25519':
                public_key, private_key = sr25519.pair_from_seed(seed_hex)
            elif crypto_type == "ed25519":
                private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
            else:
                raise ValueError('crypto_type "{}" not supported'.format(crypto_type))
            ss58_address = ss58_encode(public_key, self.ss58_format)
            keypair = Key(private_key=private_key, crypto_type=crypto_type)
        keypair.mnemonic = mnemonic
        return keypair
   
    def from_private_key(
            self, 
            private_key: Union[bytes, str],
            crypto_type: int = None
    ) -> 'Key':
        """
        Creates Key for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        crypto_type: Use KeyType.[SR25519|ED25519|ECDSA] cryptography for generating the Key
        Returns
        -------
        Key
        """
        crypto_type = self.get_crypto_type(crypto_type)
        return Key(private_key=private_key, crypto_type=crypto_type)

    def encode_signature_data(self, data: Union[ScaleBytes, bytes, str, dict]) -> bytes:
        """
        Encodes data for signing and vefiying,  converting it to bytes if necessary.
        """
        data = deepcopy(data)
        if not isinstance(data, str):
            data = python2str(data)
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x': # hex string
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()
        return data

    def resolve_signature(self, signature: Union[bytes, str]):
        if isinstance(signature,str) and signature[0:2] == '0x':
            signature = bytes.fromhex(signature[2:])
        if type(signature) is str:
            signature = bytes.fromhex(signature)
        if type(signature) is not bytes:
            raise TypeError(f"Signature should be of type bytes or a hex-string {signature}")
        return signature

    def resolve_public_key(self, address=None, public_key=None):
        if address != None:
            if is_valid_ss58_address(address):
                public_key = ss58_decode(address)
            else:
                public_key = address
        if public_key == None:
            public_key = self.public_key
        if isinstance(public_key, str) :
            if public_key.startswith('0x'):
                public_key = public_key[2:]
            public_key = bytes.fromhex(public_key)
        return public_key


    def sign(self, data: Union[ScaleBytes, bytes, str], mode='bytes') -> bytes:
        """
        Creates a signature for given data
        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format
        Returns
        -------
        signature in bytes

        """

        data = self.encode_signature_data(data)

        if self.crypto_type == "sr25519":
            signature = sr25519.sign((self.public_key, self.private_key), data)
        elif self.crypto_type == "ed25519":
            signature = ed25519_zebra.ed_sign(self.private_key, data)
        elif self.crypto_type == "ecdsa":
            signature = ecdsa_sign(self.private_key, data)
        else:
            raise Exception("Crypto type not supported")

        if mode in ['str', 'hex']:
            signature = '0x' + signature.hex()
        elif mode in ['dict', 'json']:
            signature =  {
                    'data':data.decode(),
                    'crypto_type':self.crypto_type,
                    'signature':signature.hex(),
                    'address': self.ss58_address}
        elif mode == 'bytes':
            pass
        else:
            raise ValueError(f'invalid mode {mode}')

        return signature

    def verify(self, 
               data: Union[ScaleBytes, bytes, str, dict], 
               signature: Union[bytes, str] = None,
               address : Optional[str] = None,
               public_key:Optional[str]= None, 
               max_age = None,
               **kwargs
               ) -> bool:
        """
        Verifies data with specified signature
        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format
        address: Substrate address
        public_key: public key in bytes or hex string format
        """


        if isinstance(data, dict) and  all(k in data for k in ['data','signature', 'address']):
            data, signature, address = data['data'], data['signature'], data['address']
        data = self.encode_signature_data(data)
        signature = self.resolve_signature(signature)
        public_key = self.resolve_public_key(address=address, public_key=public_key)
        if self.crypto_type == "sr25519":
            crypto_verify_fn = sr25519.verify
        elif self.crypto_type == "ed25519":
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif self.crypto_type == "ecdsa":
            crypto_verify_fn = ecdsa_verify
        else:
            raise Exception("Crypto type not supported")
        verified = crypto_verify_fn(signature, data, public_key)
        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', public_key)
        return verified
        
    def __str__(self):
        return  f'<Key(address={self.key_address} crypto_type={self.crypto_type}>'

    def from_uri(
            self, 
            suri: str, 
            crypto_type=None, 
            DEV_PHRASE = 'bottom drive obey lake curtain smoke basket hold race lonely fit walk'

    ) -> 'Key':
        """
        Creates Key for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`

        Parameters
        ----------
        suri:
        crypto_type: Use "sr25519" or "ed25519"cryptography for generating the Key

        Returns
        -------
        Key
        """
        # GET THE MNEMONIC (PHRASE) AND DERIVATION PATHS
        suri = str(suri)
        if not suri.startswith('//'):
            suri = '//' + suri
        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri
        suri_parts = re.match(r'^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri).groupdict()
        mnemonic = suri_parts['phrase']
        crypto_type = self.get_crypto_type(crypto_type)
        if crypto_type == "ecdsa":
            private_key = mnemonic_to_ecdsa_private_key(
                mnemonic=mnemonic,
                str_derivation_path=suri_parts['path'],
                passphrase=suri_parts['password']
            )
            derived_keypair = self.from_private_key(private_key, crypto_type=crypto_type)
        elif crypto_type in ["sr25519", "ed25519"]:
            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for crypto_type '{crypto_type}'")
            derived_keypair = self.from_mnemonic(mnemonic, crypto_type=crypto_type)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))
        return derived_keypair

    def from_password(self, password:str, crypto_type=None, **kwargs):
        return self.from_uri(password, crypto_type=crypto_type, **kwargs)

    def str2key(self, password:str, crypto_type=None, **kwargs):
        return self.from_password(password, crypto_type=crypto_type, **kwargs)


    # tests 

    def test_signing(self,  crypto_type=[0,1,2], data='test'):
        # at the moment, the ed25519 is not supported in the current version of pycryptodome
        if isinstance(crypto_type, list):
            return  [self.test_signing(k, data=data) for k in crypto_type]
        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        key = Key(crypto_type=crypto_type)
        sig = key.sign(data)
        assert key.verify(data,sig, key.public_key)
        return {'success':True, 'data':data, 'crypto_type' : key.crypto_type}

    def test_key_creation(self, crypto_type=None, **kwargs):
        key_name = 'test_key'
        crypto_type = self.get_crypto_type(crypto_type)
        key = self.add_key(key_name, crypto_type=crypto_type)
        assert self.key_exists(key_name, crypto_type=crypto_type), f'key {key_name} does not exist'
        self.rm_key(key_name, crypto_type=crypto_type)
        assert not self.key_exists(key_name, crypto_type=crypto_type), f'key {key_name} exists'
        return {'success':True, 'crypto_type':crypto_type}

    def test(self, crypto_type=None, **kwargs):
        test_fns = [self.test_key_creation, self.test_signing]
        results = {}
        for fn in test_fns:
            fn_name = fn.__name__
            results[fn_name] = fn(crypto_type=crypto_type, **kwargs)
        
        return results
        