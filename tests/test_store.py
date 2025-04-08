 # start of file
import unittest
import sys
import os
import time
import json
import tempfile
import shutil

# Add parent directory to path to import val
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val.store.store import Storage

class TestStorage(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage = Storage(storage_dirpath=self.temp_dir)
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_put_and_get(self):
        """Test putting and getting data"""
        test_data = {"test": "data", "nested": {"value": 123}}
        path = "test_file"
        
        # Put data
        self.storage.put(path, test_data)
        
        # Check that file exists
        self.assertTrue(self.storage.exists(path))
        
        # Get data
        retrieved_data = self.storage.get(path)
        
        # Check data integrity
        self.assertEqual(retrieved_data, test_data)
        
    def test_rm(self):
        """Test removing data"""
        test_data = {"test": "data"}
        path = "test_file_rm"
        
        # Put data
        self.storage.put(path, test_data)
        self.assertTrue(self.storage.exists(path))
        
        # Remove data
        self.storage.rm(path)
        
        # Check that file no longer exists
        self.assertFalse(self.storage.exists(path))
        
    def test_items(self):
        """Test retrieving all items"""
        # Put multiple items
        items = {
            "file1": {"id": 1, "value": "one"},
            "file2": {"id": 2, "value": "two"},
            "file3": {"id": 3, "value": "three"}
        }
        
        for path, data in items.items():
            self.storage.put(path, data)
            
        # Get all items
        all_items = self.storage.items()
        
        # Check count
        self.assertEqual(len(all_items), len(items))
        
        # Check content (order may vary)
        for item in all_items:
            self.assertIn(item["id"], [1, 2, 3])
            
    def test_paths(self):
        """Test retrieving all paths"""
        # Put multiple items
        items = {
            "path_test/file1": {"id": 1},
            "path_test/file2": {"id": 2},
            "path_test/subdir/file3": {"id": 3}
        }
        
        for path, data in items.items():
            self.storage.put(path, data)
            
        # Get all paths
        all_paths = self.storage.paths()
        
        # Check count
        self.assertEqual(len(all_paths), len(items))
        
        # Test search functionality
        subdir_paths = self.storage.paths(search="subdir")
        self.assertEqual(len(subdir_paths), 1)
        
    def test_item_age(self):
        """Test item age functionality"""
        path = "age_test_file"
        self.storage.put(path, {"test": "data"})
        
        # Get age
        ages = self.storage.path2age()
        
        # Age should be very small (just created)
        item_path = self.storage.get_item_path(path)
        self.assertLess(ages[item_path], 1.0)  # Less than 1 second old
        
    def test_max_age(self):
        """Test max age parameter for get"""
        path = "max_age_test"
        test_data = {"test": "data"}
        default_data = {"default": "value"}
        
        # Put data
        self.storage.put(path, test_data)
        
        # Get with max_age=None (should return data)
        data1 = self.storage.get(path, default=default_data, max_age=None)
        self.assertEqual(data1, test_data)
        
        # Get with very small max_age (should return default)
        time.sleep(0.1)  # Ensure some time has passed
        data2 = self.storage.get(path, default=default_data, max_age=0.05)
        self.assertEqual(data2, default_data)

if __name__ == '__main__':
    unittest.main()
