from transformers import PreTrainedTokenizerFast

class CustomTokenizer(PreTrainedTokenizerFast):
    # add custom token attributes to the SPECIAL_TOKENS_ATTRIBUTES
    SPECIAL_TOKENS_ATTRIBUTES = PreTrainedTokenizerFast.SPECIAL_TOKENS_ATTRIBUTES + [
        "blank_token", 
        "detect_lang_token", 
        "bem_lang_token", 
        "eng_lang_token", 
        "fra_lang_token", 
        "fon_lang_token"
    ]
    
    # add a property for each custom token for cleaner access
    @property
    def blank_token_id(self):
        return self.convert_tokens_to_ids(self.blank_token)
    
    @property
    def detect_lang_token_id(self):
        return self.convert_tokens_to_ids(self.detect_lang_token)
    
    @property
    def bem_lang_token_id(self):
        return self.convert_tokens_to_ids(self.bem_lang_token)
    
    @property
    def eng_lang_token_id(self):
        return self.convert_tokens_to_ids(self.eng_lang_token)
    
    @property
    def fra_lang_token_id(self):
        return self.convert_tokens_to_ids(self.fra_lang_token)
    
    @property
    def fon_lang_token_id(self):
        return self.convert_tokens_to_ids(self.fon_lang_token)