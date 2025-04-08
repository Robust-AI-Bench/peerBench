 # start of file
#!/usr/bin/env python3
"""
Script to run model evaluations using the val framework.
"""
import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path to import val
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val import val

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model evaluations')
    
    parser.add_argument('--task', type=str, default='add',
                        help='Task to evaluate (default: add)')
    
    parser.add_argument('--provider', type=str, default='openrouter',
                        help='Model provider (default: openrouter)')
    
    parser.add_argument('--n', type=int, default=4,
                        help='Number of models to test (default: 4)')
    
    parser.add_argument('--samples', type=int, default=4,
                        help='Samples per epoch (default: 4)')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for parallel evaluation (default: 2)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: results_YYYY-MM-DD_HH-MM-SS.json)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function to run evaluations."""
    args = parse_arguments()
    
    # Create output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output = f'results_{timestamp}.json'
    
    print(f"Starting evaluation with task '{args.task}' on provider '{args.provider}'")
    print(f"Testing {args.n} models with {args.samples} samples per model")
    
    # Initialize the evaluator
    evaluator = val(
        model_provider=args.provider,
        task=args.task,
        n=args.n,
        samples_per_epoch=args.samples,
        batch_size=args.batch_size,
        background=False,
        verbose=args.verbose
    )
    
    # Run an evaluation epoch
    print("Running evaluation epoch...")
    results = evaluator.epoch()
    
    # Convert results to JSON serializable format if needed
    if hasattr(results, 'to_dict'):
        results_data = results.to_dict(orient='records')
    else:
        results_data = results
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    
    # Print summary
    if hasattr(results, 'head'):
        print("\nTop results:")
        print(results[['model', 'score']].head())
    else:
        print("\nResults:")
        print(results)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
