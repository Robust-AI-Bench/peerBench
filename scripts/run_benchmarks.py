 # start of file
#!/usr/bin/env python3
"""
Benchmark script for Val framework

This script runs benchmarks on multiple models using the Val framework
and generates reports of the results.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from val import val

def parse_args():
    parser = argparse.ArgumentParser(description='Run benchmarks using Val framework')
    parser.add_argument('--provider', type=str, default='openrouter', 
                        help='Model provider to use')
    parser.add_argument('--task', type=str, default='add',
                        help='Task to evaluate')
    parser.add_argument('--models', type=int, default=4,
                        help='Number of models to test')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for parallel evaluation')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout per model evaluation in seconds')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for results')
    return parser.parse_args()

def run_benchmark(args):
    """Run benchmark with given parameters"""
    print(f"Starting benchmark with task: {args.task}")
    print(f"Testing {args.models} models with {args.samples} samples each")
    
    start_time = time.time()
    
    # Initialize Val
    v = val(
        model_provider=args.provider,
        task=args.task,
        n=args.models,
        samples_per_epoch=args.samples,
        batch_size=args.batch_size,
        timeout=args.timeout,
        verbose=True,
        background=False
    )
    
    # Run evaluation
    results = v.epoch()
    
    # Get aggregated results
    aggregated = v.aggregate(results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        'raw_results': results.to_dict('records') if hasattr(results, 'to_dict') else results,
        'aggregated': aggregated.to_dict('records') if hasattr(aggregated, 'to_dict') else aggregated,
        'duration': duration,
        'timestamp': datetime.now().isoformat(),
        'config': vars(args)
    }

def save_results(results, args):
    """Save benchmark results to files"""
    os.makedirs(args.output, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_path = os.path.join(args.output, f'benchmark_{args.task}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create and save visualization if we have DataFrame data
    if isinstance(results['aggregated'], list) and len(results['aggregated']) > 0:
        df = pd.DataFrame(results['aggregated'])
        
        # Create bar chart of model scores
        plt.figure(figsize=(10, 6))
        plt.bar(df['model'], df['score'])
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title(f'Model Performance on {args.task} Task')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(args.output, f'benchmark_{args.task}_{timestamp}.png')
        plt.savefig(plot_path)
        
    print(f"Results saved to {args.output} directory")
    return json_path

def main():
    args = parse_args()
    print("Running Val benchmark...")
    results = run_benchmark(args)
    json_path = save_results(results, args)
    
    print("\nBenchmark Summary:")
    print(f"Task: {args.task}")
    print(f"Models tested: {args.models}")
    print(f"Samples per model: {args.samples}")
    print(f"Total duration: {results['duration']:.2f} seconds")
    print(f"Results saved to: {json_path}")

if __name__ == '__main__':
    main()
