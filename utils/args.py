import argparse
import json

def parse_config():
    parser = argparse.ArgumentParser(description='Train an autoregressive model with diffusion loss.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration JSON file.')
    args = parser.parse_args()
    
    with open(args.model_config, 'r') as f:
        config = json.load(f)
    
    return config