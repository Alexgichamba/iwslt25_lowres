from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from e2e_st.text.tokenizer import CustomTokenizer
import argparse
import json
import yaml
import os
import string

def parse_args():
    parser = argparse.ArgumentParser(description="Train a new tokenizer from a text corpus")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON or YAML)")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a JSON or YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Determine file type from extension
    _, ext = os.path.splitext(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if ext.lower() in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        elif ext.lower() == '.json':
            import json
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

def get_training_corpus(text_path: str, case_standardization: str = "", remove_punctuation: bool = True):
    with open(text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if case_standardization == "lower":
        lines = [line.lower() for line in lines]
    elif case_standardization == "upper":
        lines = [line.upper() for line in lines]

    if remove_punctuation:
        translator = str.maketrans("", "", string.punctuation)
        lines = [line.translate(translator) for line in lines]
    
    for i in range(0, len(lines), 1000):
        yield lines[i : i + 1000]

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract config values with defaults
    text_path = config.get("text_path", ".")
    vocab_size = config.get("vocab_size", 1024)
    repo_id = config.get("repo_id", "iwslt25_lowres")
    push_to_hub = config.get("push_to_hub", False)
    case_standardization = config.get("case_standardization", "")
    remove_punctuation = config.get("remove_punctuation", True)
    
    # Define all special tokens - hardcoded, not from config
    special_tokens = [
        "<|blank|>", # blank
        "<|sot|>",  # bos
        "<|eot|>",  # eos
        "<|pad|>",  # pad
        "<|unk|>",  # unk
        "<|lang|>", # detect_lang
        "<|bem|>",  # bem_lang
        "<|eng|>",  # eng_lang
        "<|fra|>",  # fra_lang
        "<|fon|>"   # fon_lang
    ]
    
    # Create tokenizer backend
    tokenizer_backend = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    
    # Set up the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train tokenizer on your corpus
    corpus = get_training_corpus(text_path, case_standardization, remove_punctuation)
    data_for_training = []
    for batch in corpus:
        data_for_training.extend(batch)
    
    tokenizer_backend.train_from_iterator(data_for_training, trainer=trainer)
    
    # Create your custom tokenizer with the trained backend
    tokenizer = CustomTokenizer(
        tokenizer_object=tokenizer_backend,
        blank_token="<|blank|>",
        bos_token="<|sot|>",
        eos_token="<|eot|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        detect_lang_token="<|lang|>",
        bem_lang_token="<|bem|>",
        eng_lang_token="<|eng|>",
        fra_lang_token="<|fra|>",
        fon_lang_token="<|fon|>"
    )
    
    print("All special tokens:", tokenizer.all_special_tokens)
    
    # Save the tokenizer
    tokenizer.save_pretrained(repo_id)
    
    # Push to Hub if requested
    if push_to_hub:
        tokenizer.push_to_hub(repo_id)
    
    print(f"Tokenizer saved to {repo_id}")

if __name__ == "__main__":
    main()