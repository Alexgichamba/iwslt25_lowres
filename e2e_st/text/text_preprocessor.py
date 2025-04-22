# e2e_st/text/text_preprocessor.py

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
            text = text.lower().strip()
        elif self.case_standardization == "upper":
            text = text.upper().strip()

        text_tokens = self.tokenizer.encode(text)

        return text_tokens

class TranslationPreprocessor(AbstractTextPreprocessor):
    def __init__(self,
                 case_standardization: Literal["lower", "upper"] = None,
                 tokenizer: CustomTokenizer = None):
        
        self.case_standardization = case_standardization
        self.tokenizer = tokenizer

    def __call__(self, transcription: str, translation: str, source_language: str, target_language: str):
        if self.case_standardization == "lower":
            transcription = transcription.lower().strip()
            translation = translation.lower().strip()
        elif self.case_standardization == "upper":
            transcription = transcription.upper().strip()
            translation = translation.upper().strip()

        start_token = [self.tokenizer.bos_token_id]
        end_token = [self.tokenizer.eos_token_id]
        detect_lang_token = [self.tokenizer.detect_lang_token_id]
        source_lang_token = [getattr(self.tokenizer, f"{source_language}_lang_token_id")]
        target_lang_token = [getattr(self.tokenizer, f"{target_language}_lang_token_id")]

        transcription_tokens = self.tokenizer.encode(transcription)
        translation_tokens = self.tokenizer.encode(translation)

        input_tokens = detect_lang_token + target_lang_token + transcription_tokens + start_token + translation_tokens
        st_target_tokens = source_lang_token + transcription_tokens + start_token + translation_tokens + end_token

        return input_tokens, st_target_tokens, transcription_tokens