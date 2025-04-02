import os
import yaml
import argparse
import wandb
import gc
from tqdm import tqdm
from dataclasses import dataclass
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from e2e_st.model.transformer import Transformer
from e2e_st.audio.audio_preprocessor import LogMelSpec
from e2e_st.text.text_preprocessor import TranscriptionPreprocessor, TranslationPreprocessor
from e2e_st.text.tokenizer import CustomTokenizer
from transformers import AutoTokenizer
from e2e_st.inference import GreedyDecoding
from e2e_st import metrics
from e2e_st.utils.schedulers import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from e2e_st.model.model_config import load_or_create_model

@dataclass
class SpecConfig:
    n_mels: int
    hop_length: int
    n_fft: int
    sample_rate: int

class STDataset(Dataset):
    def __init__(
        self,
        dataset_json: str,
        audio_base_path: str,
        model: Transformer,
        tokenizer: CustomTokenizer,
        spec_config: SpecConfig,
        source_language: str = None,
        target_language: str = None,
        case_standardization: str = None,

    ):
        """
        Dataset for Whisper fine-tuning.
        
        Parameters
        ----------
        dataset_json : str
            Path to the dataset JSON file containing audio and text mappings
        audio_base_path : str
            Absolute path for the directory containing audio files
        model : Transformer
            The transformer model to be used
        tokenizer : CustomTokenizer
            The tokenizer to be used for text processing
        spec_config : SpecConfig
            Configuration for the spectrogram
        source_language : str
            Source language for translation task
        target_language : str
            Target language for translation task
        case_standardization : str
            Case standardization for text processing
        """
        self.model = model

        # Create tokenizer
        self.tokenizer = tokenizer
        
        self.source_language = source_language
        self.target_language = target_language
        self.spectogram = LogMelSpec(
                                    n_mels=spec_config.n_mels,
                                    hop_length=spec_config.hop_length,
                                    n_fft=spec_config.n_fft,
                                    sample_rate=spec_config.sample_rate
                                    )
        self.text_preprocessor = TranslationPreprocessor(
            source_language=source_language,
            target_language=target_language,
            tokenizer=tokenizer,
            case_standardization=case_standardization
        )

        # Read json file
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)
        
        # Create a mapping of audio IDs, audio paths, transcripts and translations
        self.samples = []
        for i, item in enumerate(dataset):
            # Extract the audio filename as ID and path
            audio_id = f"{source_language}_{i}".split(".")[0]
            audio_path = os.path.join(audio_base_path, item["audio_path"])
            
            # Get transcript and translation from the json
            transcript = item[f"{source_language}_transcript"]
            translation = item[f"{target_language}_translation"]
            
            # Add the tuple to samples
            self.samples.append((audio_id, audio_path, transcript, translation))
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, audio_path, transcript, translation = self.samples[idx]
        
        # Load and preprocess audio
        mel = self.spectogram(audio_path)
        
        # Get the input and target tokens
        input_tokens, st_target_tokens, asr_target_tokens = self.text_preprocessor(transcript, translation)
        # Convert to tensors
        input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        st_target_tokens = torch.tensor(st_target_tokens, dtype=torch.long)
        asr_target_tokens = torch.tensor(asr_target_tokens, dtype=torch.long)
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "st_target_tokens": st_target_tokens,
            'asr_target_tokens': asr_target_tokens
        }


    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"] for item in batch] # (n_mels, T)
        input_tokens = [item["input_tokens"] for item in batch] # (n_tokens,)
        st_target_tokens = [item["st_target_tokens"] for item in batch] # (n_tokens,)
        asr_target_tokens = [item["asr_target_tokens"] for item in batch] # (n_tokens,)
        
        mels = torch.stack(mels)
        speech_lengths = torch.tensor([mel.size(1) for mel in mels], dtype=torch.long)
        text_lengths = torch.tensor([len(tokens) for tokens in input_tokens], dtype=torch.long)
        
        # Pad token sequences
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        st_target_tokens = pad_sequence(st_target_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        asr_target_tokens = pad_sequence(asr_target_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return {
            "mel": mels,
            "speech_lengths": speech_lengths,
            "text_lengths": text_lengths,
            "input_tokens": input_tokens,
            "st_target_tokens": st_target_tokens,
            "asr_target_tokens": asr_target_tokens
        }

class STTrainer:
    def __init__(
        self,
        model: Transformer,
        tokenizer: CustomTokenizer,
        optimizer: torch.optim.Optimizer,
        target_lang_id: int,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        label_smoothing: float = 0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,
        ctcloss_weight: float = 0.0,
        max_decoding_length: int = 256
    ):
        """
        Trainer for Whisper models.
        
        Parameters
        ----------
        model : Transformer
            The transformer model to be trained
        tokenizer : CustomTokenizer
            The tokenizer to be used for text processing
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        target_lang_id : int
            Target language ID token
        label_smoothing : float
            Label smoothing factor (default: 0.0)
        lr_scheduler : 
            Learning rate scheduler
        device : str
            Device to use for training
        use_mixed_precision : bool
            Whether to use mixed precision training (only works on CUDA)
        ctcloss_weight : float
            Weight for CTC loss in the total loss calculation
        max_decoding_length : int
            Maximum decoding length for validation
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.ctcloss_weight = ctcloss_weight

        self.ctcloss = torch.nn.CTCLoss(blank=tokenizer.pad_token_id)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, 
                                                            label_smoothing=label_smoothing)

        self.target_lang_id = target_lang_id
        self.max_decoding_length = max_decoding_length
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision and device.startswith("cuda")
        self.scaler = GradScaler(device="cuda") if self.use_mixed_precision else None

        # decoding for validation
        self.decoding =  GreedyDecoding(
                                        tokenizer=tokenizer,
                                        ctc_beam_size=2
                                        )
        
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Parameters
        ----------
        batch : dict
            Batch of data from the DataLoader
            
        Returns
        -------
        loss : float
            Loss value for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        mel = batch["mel"].to(self.device)
        input_tokens = batch["input_tokens"].to(self.device)
        st_target_tokens = batch["st_target_tokens"].to(self.device)
        asr_target_tokens = batch["asr_target_tokens"].to(self.device)
        speech_lengths = batch["speech_lengths"].to(self.device)
        text_lengths = batch["text_lengths"].to(self.device)
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
            # forward pass through the model
            dec_logits, ctc_logits, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights = self.model(mel,
                                                                                                                input_tokens,
                                                                                                                speech_lengths
                                                                                                                )
            
            ctc_logits = F.log_softmax(ctc_logits, dim=-1) # (N, T, vocab_size)
            ctc_logits = ctc_logits.permute(1, 0, 2) # (T, N, vocab_size)
            # Compute CTC loss
            ctc_loss = self.ctcloss(log_probs=ctc_logits,
                                    targets=asr_target_tokens,
                                    input_lengths=speech_lengths,
                                    target_lengths=text_lengths)
            
            dec_logits = dec_logits.view(-1, dec_logits.size(-1)) # (N*T, vocab_size)
            st_target_tokens = st_target_tokens.view(-1) # (N*T)
            # Compute cross-entropy loss
            ce_loss = self.cross_entropy_loss(dec_logits, st_target_tokens)
            # Compute total loss
            loss = ce_loss + self.ctcloss_weight * ctc_loss

        # Mixed precision backward pass and optimization
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # if the scheduler is CosineAnnealingWarmupRestarts, step it
        if isinstance(self.lr_scheduler, CosineAnnealingWarmupRestarts):
            self.lr_scheduler.step()
            
        return loss.item()
    
    def validate(self, dataloader):
        """
        Validate the model on validation data.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for validation data
            
        Returns
        -------
        metrics : dict
            Dictionary of validation metrics (wer, cer, bleu, chrF++)
        """
        self.model.eval()
        total_wer = 0
        total_cer = 0
        total_bleu = 0
        total_chrf = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                mel = batch["mel"].to(self.device)
                st_target_tokens = batch["st_target_tokens"].to(self.device)
                asr_target_tokens = batch["asr_target_tokens"]  # Keep on CPU
                speech_lengths = batch["speech_lengths"].to(self.device)
                
                batch_size = mel.size(0)
                batch_count += 1
                
                # Mixed precision evaluation
                with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
                    results = self.decoding(model=self.model,
                                        target_lang_id=self.target_lang_id,
                                        speech_features=mel,
                                        speech_lengths=speech_lengths,
                                        max_length=self.max_decoding_length)
                
                # Process target sequences properly (batch as a whole)
                st_targets = []
                asr_targets = []
                
                for i in range(batch_size):
                    # ST target
                    st_tgt = st_target_tokens[i]
                    st_tgt = st_tgt[st_tgt != self.tokenizer.pad_token_id]
                    st_tgt = st_tgt[st_tgt != self.tokenizer.eos_token_id]
                    st_targets.append(self.tokenizer.decode(st_tgt.tolist()))
                    
                    # ASR target
                    asr_tgt = asr_target_tokens[i]
                    asr_tgt = asr_tgt[asr_tgt != self.tokenizer.pad_token_id]
                    asr_targets.append(self.tokenizer.decode(asr_tgt.tolist()))
                
                # Compute metrics for this batch
                metrics = metrics.compute_metrics(st_target=st_targets,
                                        asr_target=asr_targets,
                                        st_pred=results["st_preds"],
                                        asr_pred=results["asr_preds"])
                
                # Accumulate batch-total metrics
                total_wer += metrics["wer"]
                total_cer += metrics["cer"] 
                total_bleu += metrics["bleu"]
                total_chrf += metrics["chrf"]
                
                del batch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate overall averages across batches
        return {
            "wer": total_wer / batch_count,
            "cer": total_cer / batch_count,
            "bleu": total_bleu / batch_count,
            "chrf": total_chrf / batch_count,
        }
    
    def train(self, train_dataloader, val_dataloader, num_epochs, log_interval=10, use_wandb=False, output_dir="output"):
        """
        Train the model.
        
        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader for training data
        val_dataloader : DataLoader
            DataLoader for validation data
        num_epochs : int
            Number of epochs to train for
        log_interval : int
            Interval for logging training progress
        use_wandb : bool
            Whether to log metrics to Weights & Biases
            
        Returns
        -------
        history : dict
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "wer": [],
            "cer": [],
            "bleu": [],
            "chrf": []
        }
        
        # Print mixed precision status
        if self.use_mixed_precision:
            print("Using mixed precision training (FP16)")
        else:
            print("Using full precision training (FP32)")
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                # Log batch-level metrics
                if use_wandb:
                    wandb.log({
                        "train/step": epoch * len(train_dataloader) + i,
                        "train/loss": loss,
                        "train/LR": self.optimizer.param_groups[0]['lr'] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                    })
                
                if (i + 1) % log_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss:.4f}")

                del batch
                torch.cuda.empty_cache()
                gc.collect()
            
            avg_train_loss = epoch_loss / len(train_dataloader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            history["wer"].append(val_metrics["wer"])
            history["cer"].append(val_metrics["cer"])
            history["bleu"].append(val_metrics["bleu"])
            history["chrf"].append(val_metrics["chrf"])
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Validation WER: {val_metrics['wer']:.4f}, "
                  f"Validation CER: {val_metrics['cer']:.4f}, "
                  f"Validation BLEU: {val_metrics['bleu']:.4f}, "
                  f"Validation chrF: {val_metrics['chrf']:.4f}")
            # Log validation metrics
            
            # Log epoch-level metrics
            if use_wandb:
                wandb.log({
                    "train/epoch": epoch + 1,
                    "val/wer": val_metrics["wer"],
                    "val/cer": val_metrics["cer"],
                    "val/bleu": val_metrics["bleu"],
                    "val/chrf": val_metrics["chrf"]
                })
            save_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{save_epoch}.pt")
            
            # Create checkpoint dictionary
            checkpoint = {
                "epoch": save_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "val_metrics": val_metrics
            }
            
            # Save the checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Log checkpoint as artifact to wandb if enabled
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{save_epoch}", 
                    type="model"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
            
        return history

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_from_config(config, model, tokenizer):
    
    # Data paths
    data_root = config['data']['root']
    audio_base_path = config['data']['audio_root']
    train_json = f"{data_root}/train/train.json"
    val_json = f"{data_root}/val/val.json"
    
    # Task settings
    source_language = config['task']['source_language']
    target_language = config['task']['target_language']
    
    # Training settings
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.001)
    device = config['training']['device']
    
    # Mixed precision settings (with default if not specified)
    use_mixed_precision = config['training'].get('mixed_precision', True)

    # Logging settings
    logging_interval = config['logging'].get('log_interval', 100)
    
    # WandB settings - look in both potential locations
    use_wandb = config.get('wandb', {}).get('enabled', False) or config.get('logging', {}).get('wandb', False)
    wandb_project = config.get('wandb', {}).get('project', config.get('logging', {}).get('wandb_project', 'e2e_st'))
    wandb_run_name = config.get('wandb', {}).get('run_name', config.get('logging', {}).get('wand_run_name', None))
    wandb_tags = config.get('wandb', {}).get('tags', [])
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config={
                'model': config['model']['name'],
                'source_language': source_language,
                'target_language': target_language,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
            }
        )
    
    # Output settings
    output_dir = config['output']['dir']
    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_workers = config['training'].get('num_workers', 4)

    # build spec config
    spec_config = SpecConfig(
        n_mels=config['audio']['n_mels'],
        hop_length=config['audio']['hop_length'],
        n_fft=config['audio']['n_fft'],
        sample_rate=config['audio']['sample_rate']
    )

    case_standardization = config['text'].get('case_standardization', None)
    
    # Create datasets
    train_dataset = STDataset(dataset_json=train_json,
                              audio_base_path=audio_base_path,
                              model=model,
                              tokenizer=tokenizer,
                              spec_config=spec_config,
                              source_language=source_language,
                              target_language=target_language,
                              case_standardization=case_standardization
                              )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    val_dataset = STDataset(dataset_json=val_json,
                              audio_base_path=audio_base_path,
                              model=model,
                              tokenizer=tokenizer,
                              spec_config=spec_config,
                              source_language=source_language,
                              target_language=target_language,
                              case_standardization=case_standardization
                              )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Create learning rate scheduler from config
    scheduler_name = config['training'].get('scheduler', 'CosineAnnealingWarmupRestarts')
    if scheduler_name == 'CosineAnnealingWarmupRestarts':
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=len(train_dataloader),
            max_lr=learning_rate,
            min_lr=config['training'].get('min_lr', 1e-6),
            warmup_steps=config['training'].get('warmup_steps', 100),
            gamma=config['training'].get('gamma', 0.5),
        )
    elif scheduler_name == 'WarmupReduceLROnPlateau':
        lr_scheduler = WarmupReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['training'].get('factor', 0.1),
            patience=config['training'].get('patience', 5),
            warmup_steps=config['training'].get('warmup_steps', 100),
            threshold=config['training'].get('threshold', 0.0001),
            min_lr=config['training'].get('min_lr', 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    

    # Create model trainer
    trainer = STTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        target_lang_id=tokenizer.lang2id[target_language],
        lr_scheduler=lr_scheduler,
        label_smoothing=config['training'].get('label_smoothing', 0.1),
        device=device,
        use_mixed_precision=use_mixed_precision,
        ctcloss_weight=config['training'].get('ctcloss_weight', 0.0),
        max_decoding_length=config['training'].get('max_decoding_length', 256)
    )
    # Train the model
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        log_interval=logging_interval,
        use_wandb=use_wandb,
        output_dir=output_dir
    )
    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    # Finish wandb run if enabled
    if use_wandb:
        wandb.finish()
        print("WandB run finished.")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Speech Translation model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output (overrides config)")
    args = parser.parse_args()
    
    # Initialize model, get model config and full config
    model, model_config, full_config = load_or_create_model(args.config)
    
    # Update output directory if specified
    if args.output_dir:
        full_config['output']['dir'] = args.output_dir
    
    AutoTokenizer.register("custom", None, CustomTokenizer)
    tokenizer_name = full_config['text'].get('tokenizer', "alexgichamba/iwslt25_lowres_uncased_4096")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Set vocab size in the model if necessary
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"Warning: Model vocab size ({model_config.vocab_size}) doesn't match tokenizer size ({tokenizer.vocab_size}).")
        print("Consider updating your model configuration.")
    
    # Move model to the specified device
    device = full_config['training']['device']
    model = model.to(device)
    
    # Log model info
    print(f"Model configuration: {model_config.to_dict()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Train the model
    history = train_from_config(full_config, model, tokenizer)