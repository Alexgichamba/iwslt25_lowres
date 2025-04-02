from abc import ABC, abstractmethod
from typing import Literal
from e2e_st.text.tokenizer import CustomTokenizer

class AbstractTextPreprocessor(ABC):
    @abstractmethod
    def __call__(self, text: str):
        pass

class TranscriptionPreprocessor(AbstractTextPreprocessor):
    def __init__(self,
                 case_standardization: Literal["lower", "upper"] = None,
                 tokenizer: CustomTokenizer = None):
        
        self.case_standardization = case_standardization
        self.tokenizer = tokenizer

    def __call__(self, text: str):
        if self.case_standardization == "lower":
            text = text.lower()
        elif self.case_standardization == "upper":
            text = text.upper()

        text_tokens = self.tokenizer.encode(text)

        return text_tokens

class TranslationPreprocessor(AbstractTextPreprocessor):
    def __init__(self,
                 case_standardization: Literal["lower", "upper"] = None,
                 source_language: Literal["bem", "eng", "fr", "fon"] = "eng",
                 target_language: Literal["bem", "eng", "fr", "fon"] = "eng",
                 tokenizer: CustomTokenizer = None):
        
        self.case_standardization = case_standardization
        self.source_language = source_language
        self.target_language = target_language
        self.tokenizer = tokenizer

    def __call__(self, transcription: str, translation: str):
        if self.case_standardization == "lower":
            transcription = transcription.lower()
            translation = translation.lower()
        elif self.case_standardization == "upper":
            transcription = transcription.upper()
            translation = translation.upper()

        start_token = [self.tokenizer.bos_token_id]
        end_token = [self.tokenizer.eos_token_id]
        detect_lang_token = [self.tokenizer.detect_lang_token_id]
        source_lang_token = [getattr(self.tokenizer, f"{self.source_language}_lang_token_id")]
        target_lang_token = [getattr(self.tokenizer, f"{self.target_language}_lang_token_id")]

        transcription_tokens = self.tokenizer.encode(transcription)
        translation_tokens = self.tokenizer.encode(translation)

        input_tokens = detect_lang_token + target_lang_token + transcription_tokens + start_token + translation_tokens
        st_target_tokens = source_lang_token + transcription_tokens + start_token + translation_tokens + end_token

        return input_tokens, st_target_tokens, transcription_tokens