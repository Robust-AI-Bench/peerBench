import requests
from typing import Optional

class Lighthouse:
    def __init__(self, api_key: Optional[str]=None):
        self.api_key = os.getenv('LIGHTHOUSE_API_KEY', api_key)
        assert self.api_key, 'Lighthouse API key is required'
        self.base_url = 'https://node.lighthouse.storage/api/v0'
        self.auth_url = 'https://api.lighthouse.storage/api/auth'
    def upload_file(self, file_path):
        url = f'{self.base_url}/add'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, headers=headers, files=files)
        return response.json()

    def upload_encrypted_file(self, file_path):
        url = f'{self.base_url}/add_encrypted'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, headers=headers, files=files)
        return response.json()

    def get_auth_message(self, public_key):
        url = f'{self.auth_url}/get_message'
        response = requests.get(url, params={'publicKey': public_key})
        return response.json()

    def create_api_key(self, public_key, signed_message, key_name):
        url = f'{self.auth_url}/create_api_key'
        payload = {
            'publicKey': public_key,
            'signedMessage': signed_message,
            'keyName': key_name
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    def list_files(self):
        url = f'{self.base_url}/list'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(url, headers=headers)
        return response.json()

    def file_info(self, cid):
        url = f'{self.base_url}/file_info/{cid}'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(url, headers=headers)
        return response.json()

def test_lighthouse():
    test_api_key = 'your_test_api_key'
    lh = Lighthouse(api_key=test_api_key)

    # Test file upload
    upload_response = lh.upload_file('/path/to/testfile.jpeg')
    print('Upload Response:', upload_response)

    # Test encrypted file upload
    encrypted_upload_response = lh.upload_encrypted_file('/path/to/testfile.jpeg')
    print('Encrypted Upload Response:', encrypted_upload_response)

    # Test list files
    files_list = lh.list_files()
    print('Files List:', files_list)

    # Test file info (replace with a valid CID)
    if files_list and 'data' in files_list and len(files_list['data']) > 0:
        file_cid = files_list['data'][0]['cid']
        file_details = lh.file_info(cid=file_cid)
        print('File Details:', file_details)


