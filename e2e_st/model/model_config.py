from dataclasses import dataclass, fields
from typing import Literal, Optional, Dict, Any
import torch
import os
from e2e_st.model.transformer import Transformer


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    enc_attn_dropout: float = 0.1
    dec_attn_dropout: float = 0.1
    vocab_size: int = 10000
    
    # Audio processing
    in_channels: Optional[int] = 80
    speech_embedding_type: Literal["whisper", "speech_transformer"] = "whisper"
    
    # Feed forward network type
    ff_type: Literal["transformer", "swiglu"] = "transformer"
    
    # Positional encoding
    pe_max_len: int = 3000
    
    # Tokenizer related
    padding_idx: int = 3
    
    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # Can be 'float32' or 'float16'
    use_sdpa: bool = False
    average_attn_weights: bool = True
    
    def __post_init__(self):
        """Convert device string to torch.device."""
        self.device = torch.device(self.device)
    
    def create_model(self) -> Transformer:
        """Create a Transformer model from this configuration."""
        return Transformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_enc_layers=self.n_enc_layers,
            n_dec_layers=self.n_dec_layers,
            enc_attn_dropout=self.enc_attn_dropout,
            dec_attn_dropout=self.dec_attn_dropout,
            vocab_size=self.vocab_size,
            in_channels=self.in_channels,
            speech_embedding_type=self.speech_embedding_type,
            ff_type=self.ff_type,
            pe_max_len=self.pe_max_len,
            padding_idx=self.padding_idx,
            device=self.device,
            average_attn_weights=self.average_attn_weights,
            dtype=self.dtype
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create a TransformerConfig from a dictionary."""
        # Filter out keys that aren't part of the dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        # Handle dtype conversion
        if 'dtype' in filtered_dict:
            dtype_str = filtered_dict['dtype']
            if dtype_str == 'float32':
                filtered_dict['dtype'] = torch.float32
            elif dtype_str == 'float16':
                filtered_dict['dtype'] = torch.float16
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
        print(filtered_dict)
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TransformerConfig':
        """Create a TransformerConfig from a YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model-specific configuration
        if 'model' in config_dict:
            model_config = config_dict['model']
        else:
            model_config = config_dict
        return cls.from_dict(model_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            key: getattr(self, key) 
            for key in self.__dataclass_fields__ 
            if key != 'device' or isinstance(getattr(self, key), str)
        }
    
    def save(self, path: str) -> None:
        """Save the config to a YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)


def load_or_create_model(config_path: str) -> tuple[Transformer, TransformerConfig]:
    """
    Load or create a model based on a single configuration file.
    
    Args:
        config_path: Path to a YAML configuration file
        
    Returns:
        model: The loaded or created model
        config: The configuration used
    """
    # Load full configuration
    import yaml
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract model configuration
    model_config = TransformerConfig.from_dict(full_config.get('model', {}))
    
    # Create model
    model = model_config.create_model()
    
    # Check if there's a model path specified in the config
    model_path = full_config.get('model_checkpoint', None)
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=model_config.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Loaded pre-trained model from {model_path}")
    
    return model, model_config, full_config