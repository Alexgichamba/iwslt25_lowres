# e2e_st/text/tokenizer.py

from transformers import PreTrainedTokenizerFast
import os

class CustomTokenizer(PreTrainedTokenizerFast):
    SPECIAL_TOKENS_ATTRIBUTES = PreTrainedTokenizerFast.SPECIAL_TOKENS_ATTRIBUTES + [
        "blank_token", 
        "detect_lang_token", 
        "bem_lang_token", 
        "eng_lang_token", 
        "fra_lang_token", 
        "fon_lang_token"
    ]

    @property
    def blank_token_id(self): return self.convert_tokens_to_ids(self.blank_token)
    @property
    def detect_lang_token_id(self): return self.convert_tokens_to_ids(self.detect_lang_token)
    @property
    def bem_lang_token_id(self): return self.convert_tokens_to_ids(self.bem_lang_token)
    @property
    def eng_lang_token_id(self): return self.convert_tokens_to_ids(self.eng_lang_token)
    @property
    def fra_lang_token_id(self): return self.convert_tokens_to_ids(self.fra_lang_token)
    @property
    def fon_lang_token_id(self): return self.convert_tokens_to_ids(self.fon_lang_token)

def load_custom_tokenizer(path_or_repo_id: str) -> CustomTokenizer:
    return CustomTokenizer.from_pretrained(
        path_or_repo_id,
        bos_token="<|sot|>",
        eos_token="<|eot|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        blank_token="<|blank|>",
        detect_lang_token="<|lang|>",
        bem_lang_token="<|bem|>",
        eng_lang_token="<|eng|>",
        fra_lang_token="<|fra|>",
        fon_lang_token="<|fon|>",
        additional_special_tokens=[
            "<|blank|>", "<|lang|>", "<|bem|>", "<|eng|>", "<|fra|>", "<|fon|>"
        ]
    )
