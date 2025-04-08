 # start of file
import unittest
import sys
import os
import json
import tempfile
import hashlib

# Add parent directory to path to import val
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val.utils import *

class TestUtils(unittest.TestCase):
    
    def test_sha256(self):
        """Test SHA256 hash function"""
        test_data = "test_data"
        expected_hash = hashlib.sha256(test_data.encode()).hexdigest()
        
        # Test the function
        result = sha256(test_data)
        self.assertEqual(result, expected_hash)
        
    def test_get_hash(self):
        """Test get_hash function with different modes"""
        test_data = "test_data"
        
        # Test SHA256
        sha256_hash = get_hash(test_data, mode="sha256")
        expected_sha256 = hashlib.sha256(test_data.encode()).hexdigest() + ":sha256"
        self.assertEqual(sha256_hash, expected_sha256)
        
        # Test MD5
        md5_hash = get_hash(test_data, mode="md5")
        expected_md5 = hashlib.md5(test_data.encode()).hexdigest() + ":md5"
        self.assertEqual(md5_hash, expected_md5)
        
        # Test with prefix
        prefixed_hash = get_hash(test_data, mode="sha256", add_prefix=True)
        expected_prefixed = "sha256:" + hashlib.sha256(test_data.encode()).hexdigest() + ":sha256"
        self.assertEqual(prefixed_hash, expected_prefixed)
        
    def test_str2python(self):
        """Test string to Python object conversion"""
        # Test basic types
        self.assertEqual(str2python("None"), None)
        self.assertEqual(str2python("null"), None)
        self.assertEqual(str2python("true"), True)
        self.assertEqual(str2python("false"), False)
        self.assertEqual(str2python("123"), 123)
        self.assertEqual(str2python("123.45"), 123.45)
        
        # Test lists
        self.assertEqual(str2python("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(str2python("[]"), [])
        
        # Test dictionaries
        self.assertEqual(str2python("{}"), {})
        self.assertEqual(str2python("{\"a\": 1, \"b\": 2}"), {"a": 1, "b": 2})
        
    def test_python2str(self):
        """Test Python object to string conversion"""
        # Test dictionary
        test_dict = {"a": 1, "b": 2}
        dict_str = python2str(test_dict)
        self.assertEqual(json.loads(dict_str), test_dict)
        
        # Test list
        test_list = [1, 2, 3]
        list_str = python2str(test_list)
        self.assertEqual(json.loads(list_str), test_list)
        
        # Test basic types
        self.assertEqual(python2str(123), "123")
        self.assertEqual(python2str(True), "True")
        
    def test_is_mnemonic(self):
        """Test mnemonic validation"""
        valid_mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        invalid_mnemonic = "invalid mnemonic phrase"
        
        self.assertTrue(is_mnemonic(valid_mnemonic))
        self.assertFalse(is_mnemonic(invalid_mnemonic))
        
    def test_abspath(self):
        """Test absolute path resolution"""
        path = "~/test/path"
        expanded_path = os.path.expanduser(path)
        abs_path = os.path.abspath(expanded_path)
        
        self.assertEqual(abspath(path), abs_path)
        
    def test_random_color(self):
        """Test random color generation"""
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white']
        color = random_color()
        self.assertIn(color, colors)
        
    def test_thread(self):
        """Test thread function"""
        result = []
        
        def test_fn(value):
            result.append(value)
            
        # Start a thread
        t = thread(test_fn, args=[42])
        
        # Wait for thread to complete
        t.join()
        
        # Check result
        self.assertEqual(result, [42])

if __name__ == '__main__':
    unittest.main()
