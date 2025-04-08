 # start of file
#!/usr/bin/env python3
"""
Script to set up the environment for val framework.
"""
import os
import sys
import argparse
import subprocess
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up the environment for val framework')
    
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key (optional)')
    
    parser.add_argument('--storage-path', type=str, default='~/.val',
                        help='Storage path for val data (default: ~/.val)')
    
    parser.add_argument('--install-deps', action='store_true',
                        help='Install dependencies')
    
    return parser.parse_args()

def create_directories(storage_path):
    """Create necessary directories."""
    dirs = [
        '',  # Base directory
        '/model',
        '/model/openrouter',
        '/task',
        '/task/add',
    ]
    
    for d in dirs:
        path = os.path.join(storage_path, d)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

def setup_api_key(storage_path, api_key):
    """Set up API key."""
    if api_key:
        # Save to environment file
        with open('.env', 'w') as f:
            f.write(f"OPENROUTER_API_KEY={api_key}\n")
        print("Added API key to .env file")
        
        # Save to storage
        api_path = os.path.join(storage_path, 'model/openrouter/api.json')
        try:
            with open(api_path, 'r') as f:
                keys = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            keys = []
        
        if api_key not in keys:
            keys.append(api_key)
            with open(api_path, 'w') as f:
                json.dump(keys, f)
            print(f"Added API key to {api_path}")

def install_dependencies():
    """Install required dependencies."""
    requirements = [
        'pandas',
        'requests',
        'openai',
        'python-dotenv',
        'tqdm',
        'scalecodec',
        'bip39',
        'sr25519',
        'ed25519_zebra',
        'pynacl',
        'eth-keys',
        'eth-utils',
        'ecdsa',
        'base58',
    ]
    
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
    print("Dependencies installed successfully")

def main():
    """Main function to set up the environment."""
    args = parse_arguments()
    
    # Expand storage path
    storage_path = os.path.expanduser(args.storage_path)
    
    # Create directories
    create_directories(storage_path)
    
    # Set up API key
    if args.api_key:
        setup_api_key(storage_path, args.api_key)
    
    # Install dependencies
    if args.install_deps:
        install_dependencies()
    
    print(f"\nEnvironment setup complete!")
    print(f"Storage path: {storage_path}")
    print("You can now use the val framework.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
