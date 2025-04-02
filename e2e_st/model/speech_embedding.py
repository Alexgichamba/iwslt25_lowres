from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class AbstractSpeechEmbedding(ABC):
    @abstractmethod
    def forward(self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
        
    @abstractmethod
    def __call__(self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(speech, speech_lengths)

class WhisperSpeechEmbedding(AbstractSpeechEmbedding):
    """
    This Speech Embedding layer follows Whisper's implementation as described in the paper:
    "Robust Speech Recognition via Large-Scale Weak Supervision" by Radford et al. 
    https://cdn.openai.com/papers/whisper.pdf
    """
    def __init__(self, in_channels:int, out_channels: int, strides: List[int] = [1, 2]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=strides[0], padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=strides[1], padding=1)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        
        # Calculate the total time downsampling factor
        self.time_downsampling_factor = strides[0] * strides[1]

    def forward(self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            speech: torch.Tensor of shape (batch_size, num_channels, num_frames)
            (Optional) speech_lengths: torch.Tensor of shape (batch_size) with sequence lengths
        Returns:
            embedding: torch.Tensor of shape (batch_size, num_channels, num_frames//stride)
            adjusted_lengths: torch.Tensor of shape (batch_size) with updated sequence lengths
        """
        x = self.gelu1(self.conv1(speech))
        x = self.gelu2(self.conv2(x))
        
        adjusted_lengths = None
        if speech_lengths is not None:
            # Calculate new lengths after downsampling
            adjusted_lengths = torch.ceil(speech_lengths.float() / self.time_downsampling_factor).int()
            # Ensure lengths don't exceed the actual sequence length
            adjusted_lengths = torch.clamp(adjusted_lengths, max=x.size(-1))
        
        return x, adjusted_lengths

class SpeechTransformerSpeechEmbedding(AbstractSpeechEmbedding):
    """"
    This Speech Embedding layer follows the implementation of the SpeechTransformer model as described in the paper:"
    "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Dong et al.
    https://ieeexplore.ieee.org/document/8462506
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.linear = nn.Linear(in_features=out_channels, out_features=out_channels)
        
        # Calculate the total time downsampling factor (2*2=4 for the time dimension)
        self.time_downsampling_factor = 4

    def forward(self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            speech: torch.Tensor of shape (batch_size, num_frames, num_features)
            (Optional) speech_lengths: torch.Tensor of shape (batch_size)
        Returns:
            embedding: torch.Tensor of shape (batch_size, num_frames//4, out_channels)
            adjusted_lengths: torch.Tensor of shape (batch_size) with updated sequence lengths
        """
        speech = speech.unsqueeze(1)  # Add channel dimension
        x = self.gelu1(self.conv1(speech))
        x = self.gelu2(self.conv2(x))

        batch_size, channels, new_frames, new_features = x.shape
        
        # Reshape: (batch_size, out_channels, new_frames, new_features) -> 
        # (batch_size, new_frames, new_features, out_channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Flatten the feature and channel dimensions
        # (batch_size, new_frames, new_features, out_channels) ->
        # (batch_size, new_frames, new_features * out_channels)
        x = x.view(batch_size, new_frames, new_features * channels)
        
        # Apply linear transformation
        x = self.linear(x)
        
        adjusted_lengths = None
        if speech_lengths is not None:
            # Calculate new lengths after downsampling
            adjusted_lengths = torch.ceil(speech_lengths.float() / self.time_downsampling_factor).int()
            # Ensure lengths don't exceed the actual sequence length
            adjusted_lengths = torch.clamp(adjusted_lengths, max=new_frames)
        
        return x, adjusted_lengths