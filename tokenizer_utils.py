from Encoder import encode, decode
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def handel_encode(self,text):
        return encode(self.tokenizer,text)
    def handel_decode(self,encoded,skip_special_tokens=True):
    
        return decode(self.tokenizer,encoded,skip_special_tokens)