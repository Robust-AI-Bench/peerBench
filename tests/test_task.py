 # start of file
import unittest
import sys
import os
import random

# Add parent directory to path to import val
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val.task.add.task import Task

class TestTask(unittest.TestCase):
    
    def setUp(self):
        self.task = Task()
        
    def test_sample_generation(self):
        """Test sample generation"""
        sample = self.task.sample()
        
        # Check sample structure
        self.assertIsInstance(sample, dict)
        self.assertIn('message', sample)
        self.assertIn('a', sample['message'])
        self.assertIn('b', sample['message'])
        self.assertIn('goal', sample['message'])
        self.assertIn('output_format', sample['message'])
        
        # Check that a and b are integers
        self.assertIsInstance(sample['message']['a'], int)
        self.assertIsInstance(sample['message']['b'], int)
        
    def test_sample_with_idx(self):
        """Test sample generation with fixed index for determinism"""
        idx = 42
        sample1 = self.task.sample(idx=idx)
        sample2 = self.task.sample(idx=idx)
        
        # Samples with same idx should be identical
        self.assertEqual(sample1['message']['a'], sample2['message']['a'])
        self.assertEqual(sample1['message']['b'], sample2['message']['b'])
        
    def test_forward(self):
        """Test forward function with mock model"""
        # Create a mock model function
        def mock_model(**kwargs):
            a = kwargs['message']['a']
            b = kwargs['message']['b']
            return f"<OUTPUT_JSON>{{\"y\": {a + b}}}</OUTPUT_JSON>"
        
        # Set a fixed seed for deterministic testing
        random.seed(42)
        idx = random.randint(1, 1000)
        
        # Generate a sample with the fixed index
        sample = self.task.sample(idx=idx)
        
        # Run the forward function with the mock model
        result = self.task.forward(mock_model, sample=sample)
        
        # Check the result
        self.assertIn('score', result)
        self.assertEqual(result['score'], 1.0)  # Perfect score for correct answer
        
    def test_scoring(self):
        """Test scoring function"""
        # Create test data with correct answer
        a, b = 5, 7
        correct_result = f"<OUTPUT_JSON>{{\"y\": {a + b}}}</OUTPUT_JSON>"
        incorrect_result = f"<OUTPUT_JSON>{{\"y\": {a + b + 1}}}</OUTPUT_JSON>"
        
        # Create sample data
        sample = {
            'message': {
                'a': a,
                'b': b,
                'goal': 'return a json object with the sum',
                'output_format': 'strictly as <OUTPUT_JSON>json(y:int)</OUTPUT_JSON>'
            }
        }
        
        # Test with correct result
        data = {
            'sample': {'message': sample['message']},
            'result': correct_result
        }
        scored_data = self.task.score(data)
        self.assertEqual(scored_data['score'], 1.0)
        
        # Test with incorrect result
        data = {
            'sample': {'message': sample['message']},
            'result': incorrect_result
        }
        scored_data = self.task.score(data)
        self.assertEqual(scored_data['score'], 0.0)

if __name__ == '__main__':
    unittest.main()
