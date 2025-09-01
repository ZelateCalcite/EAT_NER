import os
import sys
from typing import List
from tqdm import tqdm
from re import sub, search, finditer

try:
    from utils.data_processor import ner_data_process, out_filter
    from utils.algorithms import find_max_continuous_common_subarray
    from translate.qwen import qwen14_multi_round_cot_label, qwen14_multi_round_cot_text
except ModuleNotFoundError:
    python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print('Current path changed to: ', python_path)
    sys.path.insert(0, python_path)
    from utils.data_processor import ner_data_process, out_filter
    from translate.qwen import qwen14_multi_round_cot_label, qwen14_multi_round_cot_text


def output_re(model_output, sentence):
    if search('[\u4e00-\u9fa5]', model_output):
        zh_spans = []
        for m in finditer('[\u4e00-\u9fa5]+', model_output):
            zh_spans.append(model_output[m.start(): m.end()])
        res = ''
        for span in zh_spans:
            temp = ''.join(find_max_continuous_common_subarray(list(span), list(sentence)))
            if len(temp) > len(res):
                res = temp
        return res
    else:
        model_output = sub(r'\n', ' ', model_output)
        model_output = sub(r'\*\*', ' ', model_output)
        tokenized = list(filter(lambda x: x != '', model_output.split(' ')))
        return ' '.join(find_max_continuous_common_subarray(tokenized, sentence.split(' ')))



def qwen_cot_trans_text(model, tokenizer, language_name, sentences: List, entity_types):
    result = []
    for sentence in tqdm(sentences):
        result.append(qwen14_multi_round_cot_text(model, tokenizer, language_name, sentence, entity_types))
    return result


def qwen_cot_trans_label(model, tokenizer, language_name, eng_predictions: List, raw_sentences: List):
    translated = []
    for index in tqdm(range(len(raw_sentences)), total=len(raw_sentences)):
        temp = []
        for entity in eng_predictions[index]:
            trans = qwen14_multi_round_cot_label(model, tokenizer, language_name, list(entity.keys())[0],
                                                 raw_sentences[index])
            try:
                trans_re = output_re(trans, raw_sentences[index])
            except:
                trans_re = ''
            if trans_re and (trans_re in raw_sentences[index]):
                temp.append({
                    trans_re: list(entity.values())[0]
                })
        translated.append(temp)
    return translated


if __name__ == '__main__':
    pass
