import os
import re
import json
from ParserGPT.logger import logging
import random

class CreateData:
    def __init__(self, train_path:str, test_path:str, prompt_path:str):
        self.train_path = train_path
        self.test_path = test_path
        self.prompt_path = prompt_path

    def categorize_text(self, text):
    # Patterns to identify each category
        event_creation_pattern = re.compile(r'\bCreateCommitEventWrapper\b')
        org_chart_pattern = re.compile(r'\b(FindTeamOf|FindManager|FindReports)\b')
        combined_pattern = re.compile(r'\bCreateCommitEventWrapper\b.*\b(FindTeamOf|FindManager|FindReports)\b')

        if combined_pattern.search(text):
            return "compositional skills"
        elif org_chart_pattern.search(text):
            return "organization navigation skills"
        elif event_creation_pattern.search(text):
            return "event creation skills"
        return "N/A"
    
    def read_system_prompt(self,prompt_file_path):
        with open(prompt_file_path, 'r') as file:
            return file.read().strip()
    def extract_utterances_and_plans(self, json_file_path):
        data = []
        with open(json_file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
                
        dialogue_dict = {}

        for entry in data:
            dialogue_id = entry['dialogue_id']
            utterance = entry['utterance']
            plan = entry['plan']
            if dialogue_id not in dialogue_dict:
                dialogue_dict[dialogue_id] = []
            dialogue_dict[dialogue_id].append((utterance, plan))
        
        return dialogue_dict

    def transform_to_finetune_format(self,dialogue_dict, system_prompt):
        finetune_data = []

        for dialogue_id, utterances_and_plans in dialogue_dict.items():
            for utterance, plan in utterances_and_plans:
                user_prompt = utterance
                assistant_response = plan
                category = self.categorize_text(plan)
                formatted_data = f"<s>[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]\n\nSkill required: {category}\nPlan: {assistant_response}</s>"
                finetune_data.append({"text": formatted_data})

        return finetune_data
    
    def start(self):
        self.package_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        system_prompt = self.read_system_prompt(self.prompt_path)
        train_dict = self.extract_utterances_and_plans(self.train_path)
        test_dict = self.extract_utterances_and_plans(self.test_path)

        train_data = self.transform_to_finetune_format(train_dict, system_prompt)
        test_data = self.transform_to_finetune_format(test_dict, system_prompt)

        logging.info("Extracted utterance and plan from json")

        # Balancing training data
        ev,org,comp=[],[],[]
        for sr in train_data:
            x=sr['text'].split('[/INST]\n\n')[1]
            x=x.split('\nPlan:')[0]
            if x == 'Skill required: event creation skills':
                ev.append(sr)
            elif x == 'Skill required: compositional skills':
                comp.append(sr)
            else:
                org.append(sr)

        random.seed(40)
        random.shuffle(ev)
        ev = ev[:2000]
        datan = ev+org+comp
        random.shuffle(datan)

        with open(self.package_path+'/assets/data/modified_train.json', 'w') as file:
            json.dump(datan, file, indent=4)

        # Balancing training data
        ev,org,comp=[],[],[]
        for sr in test_data:
            x=sr['text'].split('[/INST]\n\n')[1]
            x=x.split('\nPlan:')[0]
            if x == 'Skill required: event creation skills':
                ev.append(sr)
            elif x == 'Skill required: compositional skills':
                comp.append(sr)
            else:
                org.append(sr)
        
        random.shuffle(ev)
        ev = ev[:50]
        random.shuffle(comp)
        comp = comp[:50]
        datan = ev+org+comp
        random.shuffle(datan)

        with open(self.package_path+'/assets/data/modified_test.json', 'w') as file:
            json.dump(datan, file, indent=4)
        
        logging.info("Balanced and shorten the train-test set for model and saved it in json")






