# e2e_st/text/train_tokenizer.py

from tokenizers.implementations import SentencePieceBPETokenizer
import argparse
import os
import json
import yaml
from huggingface_hub import create_repo, HfApi

def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer with proper Unicode handling")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON or YAML)")
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        if ext.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

def save_hf_compatible_files(output_dir, special_tokens):
    tokenizer_config = {
        "bos_token": "<|sot|>",
        "eos_token": "<|eot|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": special_tokens
    }

    special_tokens_map = {
        "bos_token": "<|sot|>",
        "eos_token": "<|eot|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>"
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)

def main():
    args = parse_args()
    config = load_config(args.config)

    text_path = config.get("text_path", ".")
    vocab_size = config.get("vocab_size", 1024)
    repo_id = config.get("repo_id", "tokenizer_model")
    push_to_hub = config.get("push_to_hub", False)

    special_tokens = [
        "<|blank|>", "<|sot|>", "<|eot|>", "<|pad|>", "<|unk|>",
        "<|lang|>", "<|bem|>", "<|eng|>", "<|fra|>", "<|fon|>"
    ]
    
    tokenizer = SentencePieceBPETokenizer()
    
    # Train the tokenizer - SentencePiece handles Unicode better by default
    tokenizer.train(
        files=[text_path] if isinstance(text_path, str) else text_path,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )

    # Save the tokenizer
    os.makedirs(repo_id, exist_ok=True)
    tokenizer.save_model(repo_id)
    tokenizer.save(os.path.join(repo_id, "tokenizer.json"))
    save_hf_compatible_files(repo_id, special_tokens)

    print(f"Tokenizer saved to {repo_id}")

    if push_to_hub:
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True
            )
            print(f"Created repository: {repo_id}")
        except Exception as e:
            print(f"Note: Repository creation resulted in: {e}")
            
        # Then upload the folder
        api = HfApi()
        api.upload_folder(
            folder_path=repo_id,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload tokenizer"
        )
        print(f"Pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()