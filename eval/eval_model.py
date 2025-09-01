import os
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

try:
    from eval.eval_utils import *
except ModuleNotFoundError:
    python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print('Current path changed to: ', python_path)
    sys.path.insert(0, python_path)
    from eval.eval_utils import *

INSTRUCTION = '''
The task is to label named entities from in the given sentence and the entity should be chosen in [PER, LOC, ORG].
Please answer in the format (entity type, entity).
Here is the sentence:
'''



def model_evaluation(checkpoint, state_dict, test_data, save_path=''):
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, trust_remote_code=False)
    model.load_state_dict(torch.load(state_dict), strict=False)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, trust_remote_code=False, legacy=False)

    generate_output(model, tokenizer, test_data, save_output=save_path)


if __name__ == '__main__':
    pass
