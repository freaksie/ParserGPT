import os
import sys
from ParserGPT.logger import logging,set_logger
from ParserGPT.exception import CustomException
from ParserGPT.components.generate_data import CreateData
from datasets import load_metric
from ParserGPT.components.get_model import CustomModel
from ParserGPT.pipeline.infer import Generate
import json

class Evaluate:
    '''
    Args:
        model_name: name of the finetuned llama model to be used
        api_key: api key for the openai
    
    Methods:
        evaulate: this function evaluates the model on the test set and returns the bleu score, rouge score and exact match score

    Example:
        >>> from ParserGPT.evaluate import Evaluate
        >>> eval_obj = Evaluate(model_name='llama-2-7b-chat-lora', api_key='your_api_key')
        >>> bleu_score, rouge_score, exact_match_score = eval_obj.evaulate(sel='hybrid')    
    '''
    def __init__(self, model_name:str, api_key:str=None):
        self.package_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name = model_name
        self.api_key=api_key
        self.logger = set_logger()
        logging.info("loading data")    
        if not os.path.isfile(self.package_path+'/assets/data/modified_test.json'):
            logging.info("Data missing, creating data from train/test jsonl")
            CreateData('train.canonical.jsonl', 'test.canonial.jsonl', 'promp2.txt')

        with open(self.package_path+'/assets/data/modified_test.json', 'r') as file:
            data = json.load(file)
        self.user_prompts=[]
        self.references=[]
        for i in data:
            self.references.append('Yield :'+i['text'].split('<</SYS>>\n')[1].split('[/INST]\n\n')[1].split('Plan: ')[1].split('Yield :')[1][:-4])
            self.user_prompts.append(i['text'].split('<</SYS>>\n')[1].split('[/INST]')[0])

        # Load metrics
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.exact_match_count = 0
        
    
    def evaulate(self, sel:str='hybrid'):
        '''
        Args:
            sel: select the model to be used for evaluation (hybrid, openai, or llama2)
        Returns:
            bleu_score: bleu score for the model
            rouge_score: rouge score for the model
            exact_match_score: exact match score for the model
        '''
        # Load the model
        self.model = CustomModel(self.model_name, self.api_key)
        try:
            logging.info("Initiating LLama2 evaluation")
            logging.info("Generating plan from LLama2")
            infer = Generate(self.model_name, self.api_key)

            # Generate outputs for the samples and calculate metrics
            for user_prompt, reference in zip(self.user_prompts, self.references):
               generated_plan, confidence= infer.parse(user_prompt,sel=sel)

               #check for exact match
               if generated_plan == reference: self.exact_match_count += 1

               self.bleu.add(prediction=generated_plan.split(), reference=[reference.split()])
               self.rouge.add(prediction=generated_plan, reference=reference)
            bleu_score = self.bleu.compute()
            rouge_score = self.rouge.compute()

            # Calculate Exact Match score
            exact_match_score = self.exact_match_count / len(self.user_prompts)

            logging.info("bleu Score achieved {} for {} model".format(bleu_score,sel))
            return bleu_score, rouge_score, exact_match_score

        except Exception as e:
            raise CustomException(e, sys)   
        


