import os
from ParserGPT.logger import logging
from ParserGPT.exception import CustomException
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
class CustomModel():
    '''
    This class is responsible for creating a custom model.
    '''
    def __init__(self, model:str):
        self.model_name = model
    
    def get_model(self)->tuple:
        '''
        This method returns the custom model.
        '''
        if not os.path.isdir(self.model_name):
            raise CustomException("Model named {} not found".format(self.model_name), sys)
        else:
            compute_dtype = getattr(torch, "float16")
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quant_config,
                device_map={"": 0}
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
            self.tokenizer.pad_token = '<PAD>'
            self.tokenizer.add_bos_token = False
            self.tokenizer.add_eos_token = False
            return (self.model, self.tokenizer)


# if __name__=='__main__':
#     CustomModel('../assets/llama-2-7b-chat-lora3').get_model()
