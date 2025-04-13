import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from e2e_st.model.transformer import Transformer
from e2e_st.text.tokenizer import CustomTokenizer
from typing import Optional
from torchaudio.models.decoder import cuda_ctc_decoder
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List

class AbstractDecoding(ABC):
    def __init__(self, tokenizer: CustomTokenizer, ctc_beam_size: int = 5):
        self.tokenizer = tokenizer
        self.ctc_beam_size = ctc_beam_size

    @abstractmethod
    def decode(self,
                model: Transformer,
               target_lang_ids: int,
               speech_features: torch.Tensor,
               speech_lengths: torch.Tensor,
               max_length: int
               ) -> List[str]:
        pass

class GreedyDecoding(AbstractDecoding):
    def __init__(self, tokenizer: CustomTokenizer, ctc_beam_size: int = 5):
        super().__init__(tokenizer, ctc_beam_size)
        self.ctc_beam_size = ctc_beam_size
        self.ctc_decoder = cuda_ctc_decoder(
            tokens=list(tokenizer.vocab.keys()),
            beam_size=ctc_beam_size,
            nbest=1
        )

    def decode(self,
               model: Transformer,
               target_lang_ids: int,
               speech_features: torch.Tensor,
               speech_lengths: Optional[torch.Tensor] = None,
               max_length: int = 256
               ) -> Dict[str, List[str]]:
        """
        Perform greedy decoding with ASR prompting
        
        Args:
            model: Transformer model
            target_lang_ids: Target language IDs tensor of shape (batch_size,)
            speech_features: Input speech features tensor of shape (batch_size, seq_len, n_mels)
            speech_lengths (Optional): Lengths of the input speech features tensor of shape (batch_size,)
            max_length: Maximum sequence length for generation
            
        Returns:
            results: A dictionary containing the generated sequences
                - 'st_preds': List of generated sequences
                - 'asr_preds': List of ASR hypotheses
        """
        device = model.device
        with torch.no_grad():
            # Move inputs to the same device as the model
            speech_features = speech_features.to(device)
            if speech_lengths is not None:
                speech_lengths = speech_lengths.to(device)
            # Embed the speech features
            encoder_outputs, encoder_output_lengths, _ = model.embed_speech(speech_features, speech_lengths)
            # Compute CTC logits
            ctc_logits = model.compute_ctc_logits(encoder_outputs)
            # Apply log softmax to get probabilities
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1) # (batch_size, seq_len, vocab_size)
            # Decode using CTC beam search
            ctc_decode_results = self.ctc_decoder(ctc_log_probs, encoder_output_lengths)
            best_hypothesis = []
            for result in ctc_decode_results:
                tokens_tensor = torch.tensor(result[0].tokens, dtype=torch.long, device=device)
                print(f"Decoded tokens: {self.tokenizer.decode(tokens_tensor.tolist())}")
                best_hypothesis.append(tokens_tensor)
            best_hypothesis_tokens = pad_sequence(best_hypothesis, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # Construct initial prompt as tensors of [<detect_lang_token>, <target_lang_token>, <transcript_hypothesis_tokens>, <start_token>]
            batch_size = best_hypothesis_tokens.size(0)
            prompt_tokens = torch.full((batch_size, 1), self.tokenizer.detect_lang_token_id, device=device) # (batch_size, 1)
            # concatenate the target language tokens from target_lang_ids
            target_lang_tokens = target_lang_ids.unsqueeze(1)  # (batch_size, 1)
            prompt_tokens = torch.cat((prompt_tokens, target_lang_tokens), dim=1)  # (batch_size, 2)
            # concatenate the transcript hypothesis tokens
            prompt_tokens = torch.cat((prompt_tokens, best_hypothesis_tokens), dim=1)
            # concatenate the start token
            prompt_tokens = torch.cat((prompt_tokens, torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=device)), dim=1)

            # Get the max generation length
            max_gen_length = max_length - prompt_tokens.size(1)

            # Create tensor to store output sequence
            output_tokens = prompt_tokens.clone()
            
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

            # Autoregressive decoding
            for i in range(max_gen_length):
                # Get current output sequence
                curr_output = output_tokens  # (batch_size, seq_len)
                
                # Forward pass through the model for next token prediction
                logits, _, _ = model.decode(
                    x=curr_output,
                    enc_output=encoder_outputs,
                    enc_output_lengths=encoder_output_lengths,
                    padding_idx=self.tokenizer.pad_token_id
                )
                
                # Get probabilities for the next token
                probs = F.softmax(logits[:, -1, :], dim=-1)  # (batch_size, vocab_size)
                
                # Greedy decoding - take the most probable token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch_size, 1)
                
                # Concatenate with the output sequence
                output_tokens = torch.cat((output_tokens, next_token), dim=1)  # (batch_size, seq_len + 1)
                
                eos_mask = next_token.squeeze(-1) == self.tokenizer.eos_token_id  # (batch_size,)
                finished |= eos_mask # set finished to True if EOS token is generated
                if finished.all(): break

            # Remove the initial prompt tokens from the output
            final_tokens = output_tokens[:, prompt_tokens.size(1):]  # (batch_size, seq_len - prompt_len)
            
            results = {} # a dictionary to store the results where st_preds and asr_preds are stored
            st_preds = []
            for i in range(batch_size):
                # Find index of first EOS token
                sequence = final_tokens[i]
                eos_indices = (sequence == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    # If EOS token exists, truncate sequence at first EOS
                    end_idx = eos_indices[0].item()
                    sequence = sequence[:end_idx]
                # Decode the token sequence to text
                text = self.tokenizer.decode(sequence.tolist())
                st_preds.append(text)

            asr_preds = []
            # convert best asr hypothesis to text
            for i in range(batch_size):
                asr_hypothesis = best_hypothesis_tokens[i]
                text = self.tokenizer.decode(asr_hypothesis.tolist())
                asr_preds.append(text)

            # Store results in the dictionary
            results['st_preds'] = st_preds
            results['asr_preds'] = asr_preds
                
            return results
    