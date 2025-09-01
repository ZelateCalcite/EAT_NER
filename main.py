import json, sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from utils.data_processor import ner_data_process, out_filter
    from eval.eval_model import model_evaluation
except ModuleNotFoundError:
    python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print('Current path changed to: ', python_path)
    sys.path.insert(0, python_path)
    from utils.data_processor import ner_data_process, out_filter
    from translate.cot_translate import qwen_cot_trans_text, qwen_cot_trans_label
    from eval.eval_model import model_evaluation

INSTRUCTION = '''
The task is to label named entities from in the given sentence and the entity should be chosen in [{}].
Please answer in the format (entity type, entity).
Here is the sentence:
'''


def run_qwen(dataset, lang, lang_fullname, model_name):
    raws = ner_data_process(f'./data/{dataset}/{lang}/test.txt', zh=lang in ['zh', 'ja']).values()
    texts = [i['text'] for i in raws][:]
    entity_types = ['LOCATION', 'PERSON', 'ORGANIZATION']
    INSTRUCTION.format(['PER', 'LOC', 'ORG'])
    save_model_name = '-' + (model_name.split('/')[-1].split('-')[1]).lower()

    trans_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    trans_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    test = qwen_cot_trans_text(trans_model, trans_tokenizer, lang_fullname, texts, entity_types)

    with open(f'./{dataset}_{lang}_qwen2.5{save_model_name}_cot.json', 'w',
              encoding='utf-8') as f:
        f.write(json.dumps(test))

    test_data = json.load(open(f'./{dataset}_{lang}_qwen2.5{save_model_name}_cot.json', 'r',
                               encoding='utf-8'))
    test_data = [{'instruction': INSTRUCTION, 'text': i} for i in test_data]
    model_evaluation(f'flan-t5-base',
                     f'output file .pt path',
                     test_data,
                     f'test_{dataset}_{lang}_qwen2.5{save_model_name}_cot.json')

    src = json.load(
        open(f'test_{dataset}_{lang}_qwen2.5{save_model_name}_cot.json', 'r',
             encoding='utf-8')
    )[:]

    predictions = [out_filter(p[0]) for p in src]
    test = qwen_cot_trans_label(trans_model, trans_tokenizer, lang_fullname, predictions, texts)
    with open(f'test_{dataset}_{lang}_qwen2.5{save_model_name}_cot_label.json',
              'w', encoding='utf-8') as f:
        f.write(json.dumps(test, ensure_ascii=False))


if __name__ == '__main__':
    pass
