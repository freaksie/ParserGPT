# ParserGPT

ParserGPT is a comprehensive package designed for generating and fine-tuning Llama models for building an event scheduling agent. This package includes three main components: `infer.py`, `Trainer.py`, and `Evaluate.py`, each responsible for different aspects of the model's lifecycle from training to inference and evaluation.

## Installation

To install ParserGPT, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/freaksie/ParserGPT.git
cd ParserGPT
pip install -e .
```

## Usage
### infer.py

This class is responsible for generating the output based on the input text using a finetuned Llama model or the OpenAI API.

#### Methods
    - read_system_prompt(file_name): Reads the system prompt from a file.
    - init(model, api_key): Initializes the class with the model and API key. Model should be in ParserGPT/assests/
    - generate_text(system_prompt, user_prompt, model='gpt-3.5-turbo', max_tokens=4000): Generates text using the OpenAI API.
    - parse(usr_prompt, sel): Parses the input text to generate the LISP plan. The sel argument determines the type of system used (i.e., Llama2, OpenAI, or Hybrid).

#### Example 
```
from ParserGPT.pipeline.infer import Generate
model = Generate('llama-2-7b-chat-lora3', 'sk-xxxxxxxxx')
user_prompt = 'Make an appointment with my team for next Monday at 9 am .'
model.parse(user_prompt, sel='hybrid')
```

#### Selector
    - OpenAI : This will use gpt3.5-turbo-chat to convert the utterance into LISP plan. This contains system prompt with in-context learning. system prompt available in ParserGPT/assets/prompt_gpt.txt

    - Llama2 : This will use the finetuned llama2 model (provided during intialization) to convert the utterance into LISP plan.

    - Hybrid : This will use the finetunned llama2 for atleat 80% requests and will delegate the request to openAI only if the confidence of generated text is less then 98%.

### Trainer.py
This class is responsible for fine-tuning the LLama2-chat model on a custom dataset.

#### Methods:
    - init(base_model, new_model, token, train_path, test_path): 
        - base_model : from hugging-face usually `meta-llama/Llama-2-7b-chat-hf`
        - new_model : Name of the finetuned model. It will be saved in ParserGPT/assets
        - token : Login token of hugging-face to download the base model from hub
        - train_path : Name of json file used for finetunning. Keep the json file in ParserGPT/assets/data
        - test_path : Name of json file used to test finetunned model. Keep the json file in ParserGPT/assets/data
    - start(): This method is responsible for fine-tuning the model.

#### Example 
```
from ParserGPT.pipeline.trainer import Trainer

trainer = Trainer("meta-llama/Llama-2-7b-chat-hf",
                  "llama-2-7b-chat-test",
                  'hf_xxxxxxxxxxxxxxxxx',
                  "train.canonical.jsonl",
                  "test.canonical.jsonl"
                  )
trainer.start()
```

### Evaluate.py
This class evaluates the fine-tuned model on a test set and returns various performance metrics.

#### Methods:
    - init(model_name, api_key): Initializes the class with the model name and API key. Loads the test data created during `Trainer()` and metrics.
    - evaluate(sel): Evaluates the model on the test set and returns the BLEU score, ROUGE score, and exact match score. The sel argument determines the type of system used (i.e., Llama2, OpenAI, or Hybrid).
    
#### Example:
```
from ParserGPT.pipeline.evaluate import Evaluate
Evaluate('llama-2-7b-chat-lora3').evaulate('llama2')
```


## Requirements
    - Python 3.9+
    - transformers
    - datasets
    - peft
    - trl
    - accelerate
    - torch
    - huggingface_hub


## Generated samples

1) Utterance : Ok , let 's see what 's coming up . Block out Feb 1 from 2 pm to 3 pm .

Generated Plan : Yield :output CreateCommitEventWrapper :event CreatePreflightEventWrapper :constraint Constraint[Event] :end ?= TimeAfterDateTime :dateTime DateAtTimeWithDefaults :date MD :day Number 1 :month Month FEBRUARY :time NumberPM :number Number 2 :time NumberPM :number Number 3 :start ?= DateAtTimeWithDefaults :date MD :day Number 1 :month Month FEBRUARY :time NumberPM :number Number 2


2) Utterance : Make an appointment with my team for next Monday at 9 am .

Generated Plan : Yield :output CreateCommitEventWrapper :event CreatePreflightEventWrapper :constraint Constraint[Event] :attendees AttendeeListHasPeople :people FindTeamOf :recipient toRecipient CurrentUser :start ?= DateAtTimeWithDefaults :date NextDOW :dow DayOfWeek MONDAY :time NumberAM :number Number 9


3) Utterance : Tell me who James Potter 's manager is .

Generated Plan : Yield :output FindManager :recipient Execute :intension refer extensionConstraint RecipientWithNameLike :constraint Constraint[Recipient] :name PersonName James Potter

## Contact

For any questions or inquiries, please contact Neel Vora:

- GitHub: [https://github.com/freaksie](https://github.com/freaksie)
- LinkedIn: [https://linkedin.com/in/neelvora27](https://linkedin.com/in/neelvora27)
- Email: [nrvora@lbl.gov](mailto:nrvora@lbl.gov)