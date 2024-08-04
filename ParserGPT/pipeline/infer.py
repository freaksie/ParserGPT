import os
from ParserGPT.logger import logging,set_logger
from ParserGPT.components.get_model import CustomModel
from transformers import GenerationConfig
import torch.nn.functional as F
from openai import OpenAI

class Generate():
    '''
    This class is responsible for generating the output based on the input text.

    Args:
        model (str): The name of the finetuned Llama model to be used.
        api_key (str, optional): The API key for OpenAI API. Defaults to None.

    Methods:
        read_system_prompt(file_name): Read the system prompt from a file.
        __init__(model, api_key): Initialize the class with the model and API key.
        generate_text(system_prompt, user_prompt, model='gpt-3.5-turbo', max_tokens=4000): Generate text using the OpenAI API.
        parse(usr_prompt, sel): Parse the input text to generate the LISP plan.

    Returns:
        tuple: A tuple containing the generated plan and any additional information.

    Example:
        >>> generate = Generate('llama-2-7b-chat-lora', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        >>> plan, confidence = generate.parse('I want to go to the park', 'llama2')
        >>> print(plan)
        'Yield :[walk to park][take the path][arrive]'
    '''
    def read_system_prompt(self, file_name):
        with open(file_name, 'r') as file:
            return file.read().strip()
        
    def __init__(self, model:str, api_key:str=None):
        self.package_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name = os.path.join(self.package_path+'/assets',model)
        self.api_key = api_key
        self.logger = set_logger()
        self.model,self.tokenizer = CustomModel(self.model_name).get_model()
        eos_token_id = self.tokenizer.eos_token_id
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eos_token_id,
            max_new_tokens=4000
        )
        self.client = OpenAI(api_key=self.api_key)
        self.tracker=[0,0]

    def generate_text(self,system_prompt, user_prompt, model='gpt-3.5-turbo', max_tokens=4000):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": 'Utterance: '+user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    def parse(self, usr_prompt:str, sel:str = 'hybrid') -> tuple:
        '''
        Parse the input text to generate LISP plan

        args:
            usr_prompt (str): Input text to be parsed
            sel (str): Model to be used for parsing. Can be 'llama2', 'openai' or  'hybrid'

        Example:
            >>> generate = Generate('llama-2-7b-chat-lora', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            >>> plan, confidence = generate.parse('I want to go to the park', 'llama2')
            >>> print(plan)
            'Yield :[walk to park][take the path][arrive]'
        '''

        if sel=='openai':
            self.system_prompt = self.read_system_prompt(self.package_path+'/assets/prompt_gpt.txt')
            generated_text = self.generate_text(self.system_prompt, usr_prompt).split('Plan:')[1].replace("\n", "")
            generated_plan = 'Yield :'+generated_text.split('Yield :')[1]
            logging.info('OpenAI-GPT3.5-turbo has generated actionable plan for "{}" '.format(usr_prompt))  
            return (generated_plan,None)
        
        else:
            self.system_prompt = self.read_system_prompt(self.package_path+'/assets/prompt2.txt')
            context=f"<s>[INST]<<SYS>>\n{self.system_prompt}\n<</SYS>>\n{usr_prompt}[/INST]"
            inputs = self.tokenizer(context, return_tensors='pt')
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,output_scores=True, return_dict_in_generate=True)
            generated_tokens = outputs.sequences[0]
            scores = outputs.scores
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_plan = 'Yield :'+generated_text.split('[/INST]')[1].split('Plan: ')[1].split('Yield :')[1]
            probabilities = []
            for score in scores:
                prob = F.softmax(score, dim=-1)
                probabilities.append(prob)

            # Get the probability of the generated tokens
            generated_probs = [probabilities[i][0, token].item() for i, token in enumerate(generated_tokens[len(inputs['input_ids'][0]):])]
            average_confidence = sum(generated_probs) / len(generated_probs)

            if sel=='llama2':
                return (generated_plan, average_confidence)
            else:
                if average_confidence >= 0.98:
                    self.tracker[0]+=1
                    logging.info('Llama2 has generated actionable plan for "{}" with high confidence of {}'.format(usr_prompt,average_confidence))
                    return (generated_plan, average_confidence)
                
                else:
                    logging.info('Llama2 has generated actionable plan for "{}" with low confidence of {}'.format(usr_prompt, average_confidence))
                    logging.info('Redirecting the request to OpenAI')
                    if self.tracker[1] < self.tracker[0]*.2:
                        self.system_prompt = self.read_system_prompt(self.package_path+'/assets/prompt_gpt4.txt')
                        generated_text = self.generate_text(self.system_prompt, usr_prompt).split('Plan:')[1].replace("\n", "")
                        generated_plan = 'Yield :'+generated_text.split('Yield :')[1]
                        self.tracker[1]+=1
                        logging.info('OpenAI-GPT3.5-turbo has generated actionable plan for "{}" '.format(usr_prompt))  
                        return (generated_plan,None)
                    else:
                        self.tracker[0]+=1
                        logging.info('Cannot redirect, API Hit is above threshold')
                    return (generated_plan, average_confidence)
    


# if __name__=='__main__':
#     isnt=Generate('llama-2-7b-chat-lora3')
#     print(isnt.parse("I also need a team meeting with everyone on my team"))
#     print(isnt.parse("Tell me who James Potter 's manager is ."))