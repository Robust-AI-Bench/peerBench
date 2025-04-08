 # start of file
import unittest
import os
import sys
import json
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path to import val
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val import val

class TestVal(unittest.TestCase):
    
    def setUp(self):
        self.v = val(background=False, verbose=False)
        
    def test_init(self):
        """Test initialization of val object"""
        v = val(background=False, verbose=False, batch_size=8, samples_per_epoch=4, timeout=4)
        self.assertIsNotNone(v)
        self.assertEqual(v.batch_size, 8)
        self.assertEqual(v.samples_per_epoch, 4)
        self.assertEqual(v.timeout, 4)
        
    def test_hash(self):
        """Test hash function"""
        test_data = "test_data"
        hash_result = self.v.hash(test_data)
        self.assertIsInstance(hash_result, str)
        # Test that hash is deterministic
        self.assertEqual(hash_result, self.v.hash(test_data))
        

    def test_get_key(self):
        """Test key generation"""
        key = self.v.get_key('test_key')
        self.assertIsNotNone(key)
        self.assertTrue(hasattr(key, 'sign'))
        self.assertTrue(hasattr(key, 'verify'))
        
    def test_set_task(self):
        """Test setting a task"""
        result = self.v.set_task('add')
        self.assertTrue(result['success'])
        self.assertEqual(result['name'], 'add')
        self.assertIsNotNone(result['cid'])
        
    @patch('val.val.val.eval')
    def test_epoch(self, mock_eval):
        """Test epoch execution"""
        # Setup mock
        mock_eval.return_value = {
            'model': 'test_model',
            'model_provider': 'test_provider',
            'score': 0.9,
            'sample_cid': 'test_cid'
        }
        
        # Set a task and model provider
        self.v.set_task('add')
        self.v.models = ['model1', 'model2']
        self.v.samples_per_epoch = 1
        self.v.batch_size = 1
        
        # Run epoch
        results = self.v.epoch()
        
        # Check results
        self.assertIsNotNone(results)
        self.assertEqual(mock_eval.call_count, 2)  # Called for each model
        
    def test_sample(self):
        """Test sample generation"""
        self.v.set_task('add')
        sample = self.v.sample()
        self.assertIsInstance(sample, dict)
        self.assertIn('message', sample)
        self.assertIn('a', sample['message'])
        self.assertIn('b', sample['message'])
        
    def test_cid(self):
        """Test content ID generation"""
        test_data = {"test": "data"}
        cid = self.v.cid(test_data)
        self.assertIsInstance(cid, str)
        # Test deterministic
        self.assertEqual(cid, self.v.cid(test_data))
        
    @patch('val.val.thread')
    def test_background(self, mock_thread):
        """Test background process"""
        v = val(background=True, verbose=False)
        mock_thread.assert_called_once()

if __name__ == '__main__':
    unittest.main()
