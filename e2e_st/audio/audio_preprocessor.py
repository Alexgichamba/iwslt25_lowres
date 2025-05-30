# e2e_st/audio/audio_preprocessor.py
import torch
import torchaudio
from abc import ABC, abstractmethod

class AbstractAudioPreprocessor(ABC):
    @abstractmethod
    def __call__(self, waveform_path: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_audio_length(self, audio_path: str) -> float:
        """
        Args:
            audio_path, str: Path to the audio file.
        Returns:
            length, float: The length of the audio in seconds
        """
        pass

class LogMelSpec(AbstractAudioPreprocessor):
    "Compute log mel spectrogram from waveform"
    def __init__(self, n_mels: int, hop_length: int, n_fft: int, sample_rate: int):
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate  

        self.waveform_to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft
        )

        self.feature_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform_path: str) -> torch.Tensor:
        """
        Args:
            waveform_path, str: Path to the audio file.
        Returns:
            waveform, torch.Tensor: The log mel spectrogram of shape (n_mels, T)
        Load raw audio waveform from file, resample if necessary, and compute log mel spectrogram
        
        """
        waveform, sample_rate = torchaudio.load(waveform_path)
        if sample_rate != self.sample_rate:
            print(f"Resampling from {sample_rate} to {self.sample_rate}")
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
        # compute mel spectrogram
        mel_spectrogram = self.waveform_to_mel_spectrogram(waveform)
        # convert the amplitude to decibels
        mel_spectrogram = self.feature_to_db(mel_spectrogram)
        return mel_spectrogram.squeeze(0)
    
    def get_audio_length(self, audio_path):
        """Get audio length in seconds without computing the full spectrogram"""
        
        # Load audio metadata only (faster than loading the full audio)
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate

class RawAudio(AbstractAudioPreprocessor):
    "Load raw audio waveform"
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def __call__(self, waveform_path: str) -> torch.Tensor:
        """
        Args:
            waveform_path, str: Path to the audio file.
        Returns:
            waveform, torch.Tensor: The loaded audio waveform of shape (T)  
        Load raw audio waveform from file and resample if necessary
        
        """
        waveform, sample_rate = torchaudio.load(waveform_path, normalize=True)
        if sample_rate != self.sample_rate:
            print(f"Resampling from {sample_rate} to {self.sample_rate}")
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
        return waveform.squeeze(0)
    
    def get_audio_length(self, audio_path):
        """Get audio length in seconds without computing the full spectrogram"""
        
        # Load audio metadata only (faster than loading the full audio)
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate