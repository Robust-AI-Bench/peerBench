
import datasets
import random

class Task:
    description = 'Multiple choice math problems from MATH-500 dataset'
    features = ['url', 'name', 'score']

    def __init__(self, fn='info', params=None, split='test'):
        """
        Initialize the task with a function name (string)
        and optional parameters (dictionary).
        """
        self.fn = fn
        self.params = params or {}
        
        # Load the math500 dataset from Hugging Face
        self.dataset = datasets.load_dataset('HuggingFaceH4/MATH-500', split=split)
        
        # Default number of multiple choice options
        self.num_choices = self.params.get('num_choices', 4)

    def create_multiple_choice(self, sample, num_choices=None):
        """
        Create a multiple choice problem from a sample.
        
        Args:
            sample: The sample containing the question and answer
            num_choices: Number of choices to generate (including correct answer)
            
        Returns:
            A dictionary with the question, choices, and correct answer index
        """
        if num_choices is None:
            num_choices = self.num_choices
            
        # Ensure num_choices is at least 2 and not more than available samples
        num_choices = max(2, min(num_choices, len(self.dataset)))
        # Get the correct answer
        correct_answer = sample['answer']
        
        # Get random samples for incorrect answers
        incorrect_samples = []
        dataset_indices = list(range(len(self.dataset)))
        current_idx = self.dataset.index(sample) if hasattr(self.dataset, 'index') else -1
        
        if current_idx >= 0:
            dataset_indices.remove(current_idx)
            
        # Randomly select other samples for incorrect answers
        selected_indices = random.sample(dataset_indices, min(num_choices - 1, len(dataset_indices)))
        incorrect_samples = [self.dataset[idx]['answer'] for idx in selected_indices]
        
        # Create choices with the correct answer at a random position
        choices = incorrect_samples.copy()
        correct_idx = random.randint(0, num_choices - 1)
        choices.insert(correct_idx, correct_answer)
        choices = choices[:num_choices]  # Ensure we have exactly num_choices
        
        # Create the multiple choice question
        question = sample['problem']
        
        return {
            'question': question,
            'choices': choices,
            'correct_idx': correct_idx,
            'original_sample': sample
        }

    def sample(self, sample=None):
        """
        Grab a data sample from math500 at a given index and convert it to multiple choice.
        """
        idx = None
        if isinstance(sample, int):
            idx = sample   
        if isinstance(sample, dict):
            return sample
        
        if idx is None:
            idx = random.randint(0, len(self.dataset) - 1)
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range for the math500 dataset.")
        
        sample = self.dataset[idx]
        
        # Convert to multiple choice format
        return self.create_multiple_choice(sample, self.num_choices)
        
    def forward(self, model, sample=None, **call_params):
        """
        Grab a data sample from math500 at a given index, convert to multiple choice,
        then call the model and score the response.
        """
        # Get the multiple choice sample
        mc_sample = self.sample(sample)
        
        # Format the question with choices for the model
        formatted_question = mc_sample['question'] + "\n\nChoices:\n"
        for i, choice in enumerate(mc_sample['choices']):
            formatted_question += f"{chr(65+i)}. {choice}\n"
        
        # Get model response
        response = model(message=formatted_question)
        
        # Basic scoring logic - check if the model selected the correct choice
        # This is a simple implementation and might need to be refined
        correct_number = chr(65 + mc_sample['correct_idx'])
        score = 0
        
        # Check if the correct letter appears in the response
        if correct_number in response:
            score = 1
        
        result = {
            'question': mc_sample['question'],
            'choices': mc_sample['choices'],
            'correct_idx': mc_sample['correct_idx'],
            'correct_answer': mc_sample['choices'][mc_sample['correct_idx']],
            'model_response': response
        }
        
        return {
            'score': score,
            'result': result,
            'params': call_params
        }
