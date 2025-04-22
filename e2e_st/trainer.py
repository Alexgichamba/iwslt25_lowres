# e2e_st/trainer.py
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
from torch.amp import GradScaler
from torch import autocast
from e2e_st.model.transformer import Transformer
from e2e_st.audio.audio_preprocessor import LogMelSpec
from e2e_st.text.text_preprocessor import TranslationPreprocessor
from e2e_st.text.tokenizer import CustomTokenizer
from transformers import AutoTokenizer
from e2e_st.inference import GreedyDecoding
from e2e_st import metrics
from e2e_st.utils.schedulers import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from e2e_st.model.model_config import load_or_create_model
from e2e_st.utils.samplers import DurationBucketSampler

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
        tokenizer: CustomTokenizer,
        spec_config: SpecConfig,
        case_standardization: str = None,

    ):
        """
        Dataset for Whisper fine-tuning.
        
        Parameters
        ----------
        dataset_json : str
            Path to the dataset JSON file containing audio and text mappings
        tokenizer : CustomTokenizer
            The tokenizer to be used for text processing
        spec_config : SpecConfig
            Configuration for the spectrogram
        case_standardization : str
            Case standardization for text processing
        """

        # Create tokenizer
        self.tokenizer = tokenizer

        self.spectogram = LogMelSpec(
                                    n_mels=spec_config.n_mels,
                                    hop_length=spec_config.hop_length,
                                    n_fft=spec_config.n_fft,
                                    sample_rate=spec_config.sample_rate
                                    )
        self.text_preprocessor = TranslationPreprocessor(
            tokenizer=tokenizer,
            case_standardization=case_standardization
        )

        # Read the json file
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)
        
        # Create a mapping of audio paths, transcripts and translations
        self.samples = []
        for item in dataset:
            entry = dataset[item]
            audio_path = entry["audio_path"]
            transcript = entry["transcript"]
            translation = entry["translation"]
            source_language = entry["source_language"]
            target_language = entry["target_language"]
            self.samples.append((audio_path, transcript, translation, source_language, target_language))
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, transcript, translation, source_language, target_language = self.samples[idx]
        
        # Load and preprocess audio
        mel = self.spectogram(audio_path)
        
        # Get the input and target tokens
        input_tokens, st_target_tokens, asr_target_tokens = self.text_preprocessor(transcription=transcript,
                                                                                  translation=translation,
                                                                                  source_language=source_language,
                                                                                  target_language=target_language
                                                                                  )
        # Convert to tensors
        input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        st_target_tokens = torch.tensor(st_target_tokens, dtype=torch.long)
        asr_target_tokens = torch.tensor(asr_target_tokens, dtype=torch.long)

        source_language = getattr(self.tokenizer, f"{source_language}_lang_token_id")
        target_language = getattr(self.tokenizer, f"{target_language}_lang_token_id")
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "st_target_tokens": st_target_tokens,
            'asr_target_tokens': asr_target_tokens,
            "source_language": source_language,
            "target_language": target_language
        }


    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"].T for item in batch] # (T, n_mels)
        input_tokens = [item["input_tokens"] for item in batch] # (n_tokens,)
        st_target_tokens = [item["st_target_tokens"] for item in batch] # (n_tokens,)
        asr_target_tokens = [item["asr_target_tokens"] for item in batch] # (n_tokens,)
        source_languages = [torch.tensor(item["source_language"], dtype=torch.long) for item in batch] # (1,)
        target_languages = [torch.tensor(item["target_language"], dtype=torch.long) for item in batch] # (1,)

        speech_lengths = torch.tensor([mel.size(0) for mel in mels], dtype=torch.long)
        asr_text_lengths = torch.tensor([len(tokens) for tokens in asr_target_tokens], dtype=torch.long)

        # Pad token sequences
        mels = pad_sequence(mels, batch_first=True, padding_value=0) # (B, T, n_mels)
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        st_target_tokens = pad_sequence(st_target_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        asr_target_tokens = pad_sequence(asr_target_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # transpose mels to (B, n_mels, T)
        mels = mels.permute(0, 2, 1)

        source_languages = torch.stack(source_languages, dim=0) # (B, 1)
        target_languages = torch.stack(target_languages, dim=0) # (B, 1)
        
        return {
            "mel": mels,
            "speech_lengths": speech_lengths,
            "text_lengths": asr_text_lengths,
            "input_tokens": input_tokens,
            "st_target_tokens": st_target_tokens,
            "asr_target_tokens": asr_target_tokens,
            "source_languages": source_languages,
            "target_languages": target_languages
        }

class STTrainer:
    def __init__(
        self,
        model: Transformer,
        tokenizer: CustomTokenizer,
        optimizer: torch.optim.Optimizer,
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

        self.ctcloss = torch.nn.CTCLoss(blank=tokenizer.blank_token_id, zero_infinity=True)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, 
                                                            label_smoothing=label_smoothing)

        self.max_decoding_length = max_decoding_length
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler()

        # decoding for validation
        self.decoding =  GreedyDecoding(
                                        tokenizer=tokenizer,
                                        ctc_beam_size=2
                                        )
        print(f"Using device: {device}")
        
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
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # forward pass through the model
            enc_output, enc_lengths, enc_attn_weights = self.model.embed_speech(mel, speech_lengths)
            # decoder
            dec_logits, dec_self_attn_weights, dec_cross_attn_weights = self.model.decode(x=input_tokens,
                                                                                        enc_output=enc_output,
                                                                                        enc_output_lengths=enc_lengths,
                                                                                        padding_idx=self.model.padding_idx,
                                                                                        average_attn_weights=self.model.average_attn_weights
                                                                                        )
            # CTC head
            ctc_logits = self.model.compute_ctc_logits(enc_output)
            
            ctc_logits = F.log_softmax(ctc_logits, dim=-1) # (N, T, vocab_size)
            ctc_logits = ctc_logits.permute(1, 0, 2) # (T, N, vocab_size)
            # Compute CTC loss
            ctc_loss = self.ctcloss(log_probs=ctc_logits,
                                    targets=asr_target_tokens,
                                    input_lengths=enc_lengths,
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
            
        return loss.item(), ce_loss.item(), ctc_loss.item()
    
    def validate(self, dataloader, epoch=None, output_dir=None):
        """
        Validate the model on validation data using batched processing.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for validation data
        epoch : int, optional
            Current epoch number (for file naming)
        output_dir : str, optional
            Directory to save prediction files
                
        Returns
        -------
        metrics : dict
            Dictionary of validation metrics (wer, cer, bleu, chrF++)
        """
        self.model.eval()
        
        # Collect all predictions and targets, but process in batches
        all_st_preds = []
        all_st_targets = []
        all_asr_preds = []
        all_asr_targets = []
        
        with torch.no_grad():
            # Add tqdm progress bar for validation
            for batch in tqdm(dataloader, desc="Validating"):
                mel = batch["mel"].to(self.device)
                st_target_tokens = batch["st_target_tokens"].to(self.device)
                asr_target_tokens = batch["asr_target_tokens"]  # Keep on CPU
                speech_lengths = batch["speech_lengths"].to(self.device)
                # get target languages
                target_lang_ids = batch["target_languages"].to(self.device)
                
                batch_size = mel.size(0)
                
                # Mixed precision evaluation
                with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
                    results = self.decoding.decode(model=self.model,
                                    target_lang_ids=target_lang_ids,
                                    speech_features=mel,
                                    speech_lengths=speech_lengths,
                                    max_length=self.max_decoding_length)
                
                # Process target sequences properly (batch as a whole)
                for i in range(batch_size):
                    # ST target
                    st_tgt = st_target_tokens[i]
                    st_tgt = st_tgt[st_tgt != self.tokenizer.pad_token_id]
                    st_tgt = st_tgt[st_tgt != self.tokenizer.eos_token_id]
                    all_st_targets.append(self.tokenizer.decode(st_tgt.tolist()))
                    
                    # ASR target
                    asr_tgt = asr_target_tokens[i]
                    asr_tgt = asr_tgt[asr_tgt != self.tokenizer.pad_token_id]
                    all_asr_targets.append(self.tokenizer.decode(asr_tgt.tolist()))
                
                # Add predictions for this batch
                all_st_preds.extend(results["st_preds"])
                all_asr_preds.extend(results["asr_preds"])

                # Free memory after processing each batch
                del mel, st_target_tokens, speech_lengths, results, batch
                torch.cuda.empty_cache()
        
        # Write predictions to files if epoch and output_dir are provided
        if epoch is not None and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create prediction files for this epoch with side-by-side format
            st_file = os.path.join(output_dir, f"epoch_{epoch}_st_results.txt")
            asr_file = os.path.join(output_dir, f"epoch_{epoch}_asr_results.txt")
            
            # Write ST predictions and targets side by side
            with open(st_file, 'w', encoding='utf-8') as f:
                f.write("TARGET || PREDICTION\n")
                f.write("-" * 80 + "\n")  # Separator line
                for target, pred in zip(all_st_targets, all_st_preds):
                    f.write(f"{target} || {pred}\n")
                    
            # Write ASR predictions and targets side by side
            with open(asr_file, 'w', encoding='utf-8') as f:
                f.write("TARGET || PREDICTION\n")
                f.write("-" * 80 + "\n")  # Separator line
                for target, pred in zip(all_asr_targets, all_asr_preds):
                    f.write(f"{target} || {pred}\n")
                    
            print(f"Saved validation results for epoch {epoch} to {output_dir}")
        
        corpus_metrics = metrics.compute_metrics(
            st_target=all_st_targets,
            asr_target=all_asr_targets,
            st_pred=all_st_preds,
            asr_pred=all_asr_preds,
            corpus_level=True  # Get true corpus-level metrics
        )
        
        gc.collect()
        return corpus_metrics
    
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
        output_dir : str
            Directory to save model checkpoints and validation predictions
            
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
        
        # Create a subdirectory for validation predictions
        preds_dir = os.path.join(output_dir, "val_predictions")
        os.makedirs(preds_dir, exist_ok=True)
        
        # Print mixed precision status
        if self.use_mixed_precision:
            print("Using mixed precision training (FP16)")
        else:
            print("Using full precision training (FP32)")
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                loss, ce_loss, ctc_loss = self.train_step(batch)
                epoch_loss += loss
                
                # Log batch-level metrics
                if use_wandb:
                    wandb.log({
                        "train/ctc_loss": ctc_loss,
                        "train/ce_loss": ce_loss,
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
            
            # Validation - pass the epoch number and predictions directory
            val_metrics = self.validate(val_dataloader, epoch=epoch+1, output_dir=preds_dir)
            history["wer"].append(val_metrics["wer"])
            history["cer"].append(val_metrics["cer"])
            history["bleu"].append(val_metrics["bleu"])
            history["chrf"].append(val_metrics["chrf"])
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                f"Validation WER: {val_metrics['wer']:.4f}, "
                f"Validation CER: {val_metrics['cer']:.4f}, "
                f"Validation BLEU: {val_metrics['bleu']:.4f}, "
                f"Validation chrF: {val_metrics['chrf']:.4f}")
            
            # Log epoch-level metrics
            if use_wandb:
                wandb.log({
                    "train/epoch": epoch + 1,
                    "val/wer": val_metrics["wer"],
                    "val/cer": val_metrics["cer"],
                    "val/bleu": val_metrics["bleu"],
                    "val/chrf": val_metrics["chrf"]
                })
                
                # Log validation prediction files as artifacts if wandb is enabled
                val_preds_artifact = wandb.Artifact(
                    name=f"val-predictions-{epoch+1}", 
                    type="predictions"
                )
                val_preds_artifact.add_file(os.path.join(preds_dir, f"epoch_{epoch+1}_st_results.txt"))
                val_preds_artifact.add_file(os.path.join(preds_dir, f"epoch_{epoch+1}_asr_results.txt"))
                wandb.log_artifact(val_preds_artifact)
                
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
    train_json = f"{data_root}/train.json"
    val_json = f"{data_root}/valid.json"
    
    # Task settings
    language_pairs = config['language_pairs']
    source_languages = [pair['source'] for pair in language_pairs]
    target_languages = [pair['target'] for pair in language_pairs]
  
    
    # Training settings
    validation_batch_size = config['training']['validation_batch_size']
    batch_duration = config['training']['batch_duration']
    bucket_length_multiplier = config['training'].get('bucket_length_multiplier', 1.5)
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    cycle_length = config['training'].get('cycle_length', 1)
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
                'source_languages': source_languages,
                'target_languages': target_languages,
                'batch_duration': batch_duration,
                'learning_rate': learning_rate,
                'tokenizer': config['text'].get('tokenizer', "alexgichamba/iwslt25_lowres_uncased_4096"),
                'ff_type': config['model'].get('ff_type', 'linear'),
                'scheduler': config['training'].get('scheduler', 'CosineAnnealingWarmupRestarts'),
                'ctcloss_weight': config['training'].get('ctcloss_weight', 0.0),
                'learning_rate': learning_rate
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
    train_dataset = STDataset(
                            dataset_json=train_json,
                            tokenizer=tokenizer,
                            spec_config=spec_config,
                            case_standardization=case_standardization
                            )
                              
    
    # Create samplers
    train_sampler = DurationBucketSampler(
        dataset=train_dataset,
        target_duration=batch_duration,
        bucket_length_multiplier=bucket_length_multiplier,
        shuffle=True,
        shuffle_buckets=True
    )

    val_dataset = STDataset(
                            dataset_json=val_json,
                            tokenizer=tokenizer,
                            spec_config=spec_config,
                            case_standardization=case_standardization
                        )
    
    # Create DataLoader with batch_sampler instead of batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # Use batch_sampler instead of batch_size
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
                              
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create learning rate scheduler from config
    scheduler_name = config['training'].get('scheduler', 'CosineAnnealingWarmupRestarts')
    if scheduler_name == 'CosineAnnealingWarmupRestarts':
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(len(train_dataloader) * cycle_length),
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
        lr_scheduler=lr_scheduler,
        label_smoothing=config['training'].get('label_smoothing', 0.1),
        device=device,
        use_mixed_precision=use_mixed_precision,
        ctcloss_weight=config['training'].get('ctcloss_weight', 0.0),
        max_decoding_length=config['training'].get('max_decoding_length', 256)
    )

    print(f"Using duration-based batching with target duration: {batch_duration}s")
    print(f"Number of training batches: {len(train_sampler)}")
    # Training batch utterance stats
    batch_sizes = [len(batch) for batch in train_sampler.all_batches]
    print(f"Training batches: {len(batch_sizes)}")
    print(f"  - Min utterances per batch: {min(batch_sizes)}")
    print(f"  - Max utterances per batch: {max(batch_sizes)}")
    print(f"  - Avg utterances per batch: {sum(batch_sizes)/len(batch_sizes):.1f}")

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
    args = parser.parse_args()
    
    # Initialize model, get model config and full config
    model, model_config, full_config = load_or_create_model(args.config)
    print(model)
    AutoTokenizer.register("custom", None, CustomTokenizer)
    tokenizer_name = full_config.get('tokenizer', "alexgichamba/iwslt25_uncased_4096")
    print(f"Loading tokenizer: {tokenizer_name}")
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    
    # Train the model
    history = train_from_config(full_config, model, tokenizer)