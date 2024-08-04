import os
import sys
from ParserGPT.logger import logging
from ParserGPT.exception import CustomException
from ParserGPT.components.generate_data import CreateData
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
from accelerate import Accelerator
class Trainer:
    '''
    This class is responsible for finetuning LLama2-chat model on custom dataset
    Args:
        base_model (str): The base model to be used for finetuning
        new_model (str): The name of the new model to be created
        token (str): The token to be used for login
        train_path (str): The path to the training data
        test_path (str): The path to the testing data
    
    Returns:
        None
    
    methods:
        start(): This method is responsible for finetuning the model
    '''
    def __init__(self, base_model:str, new_model:str, token:str, train_path:str, test_path:str):
        self.base_model = base_model
        self.token = token
        self.package_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.new_model=os.path.join(self.package_path+'/assets',new_model)
        self.train_path=os.path.join(self.package_path+'/assets/data',train_path)
        self.test_path=os.path.join(self.package_path+'/assets/data',test_path)
        self.prompt_path=os.path.join(self.package_path+'/assets', 'prompt2.txt')

    def start(self):
        '''
        This method is responsible for finetuning the model and saves the new model and checkpoints at assets/
        
        Returns:
            None
        '''
        logging.info("Starting model finetuning")
        try:
            logging.info("Initiating model finetuning")
            
            #create dataset
            CreateData(self.train_path, self.test_path, self.prompt_path).start()
            
            # Load the dataset
            dataset = load_dataset('json', data_files={'train': self.package_path+'/assets/data/modified_train.json', 
                                                'validation': self.package_path+'/assets/data/modified_test.json'})
            compute_dtype = getattr(torch, "float16")
            logging.info("Dataset loaded")

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,  #Try doing flagging
            )

            #login to hugging face
            login(self.token)
            accelerator = Accelerator()


            model = AutoModelForCausalLM.from_pretrained(  
                self.base_model,
                quantization_config=quant_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, use_fast=False)
            tokenizer.pad_token = '<PAD>'
            tokenizer.add_bos_token = False
            tokenizer.add_eos_token = False


            peft_params = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
            )

            training_params = TrainingArguments(
                output_dir=self.package_path+"/assets/results",
                num_train_epochs=1,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                eval_strategy = "steps",
                optim="paged_adamw_32bit",
                save_steps=100,
                logging_steps=100,
                learning_rate=2e-4,
                weight_decay=0.001,
                fp16=False,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="linear", #warmup
                report_to="tensorboard",
                load_best_model_at_end=True,
                save_total_limit=5,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                gradient_checkpointing = True
            )

            finetunner = SFTTrainer(
                model=model,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                peft_config=peft_params,
                dataset_text_field="text",
                max_seq_length=None,
                tokenizer=tokenizer,
                args=training_params,
                packing=False,
            )
            finetunner = accelerator.prepare(finetunner)
            finetunner.train()

            finetunner.model.save_pretrained(self.new_model)
            finetunner.tokenizer.save_pretrained(self.new_model)

            logging.info("Model finetuning completed successfully")
        except Exception as e:
            logging.error("Error while finetuning model")
            raise CustomException(e,sys)