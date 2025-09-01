import os
import sys
import re
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support

try:
    from utils.algorithms import find_sublist_indices
    from utils.data_processor import ner_data_process
except ModuleNotFoundError:
    python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print('Current path changed to: ', python_path)
    sys.path.insert(0, python_path)
    from utils.algorithms import find_sublist_indices
    from utils.data_processor import ner_data_process


def generate_result(_tokenizer, _model, _prompts: [str], _device='cuda'):
    _inputs = _tokenizer(_prompts, return_tensors='pt', padding=True).to(_device)
    _model.to(_device)
    _outputs = _model.generate(input_ids=_inputs['input_ids'], attention_mask=_inputs['attention_mask'],
                               do_sample=False, max_new_tokens=128)
    return _tokenizer.batch_decode(_outputs, skip_special_tokens=True)


def generate_output(_model, _tokenizer, _data, save_output=''):
    _sentences = ['{}\n{}'.format(i['instruction'], i['text']) for i in _data]
    _outs = []
    for seq in tqdm(_sentences):
        _outs.append(generate_result(_tokenizer, _model, seq, _device='cuda'))
    if save_output:
        if os.path.isfile(save_output):
            with open(save_output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(_outs, ensure_ascii=False) + '\n')
        else:
            with open(save_output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(_outs, ensure_ascii=False) + '\n')

    return _outs


def label_to_seq(input_dict, zh=False):
    text = input_dict['text']
    if zh:
        text = re.sub(r' ', '#', text)
        seq = list(text)
        label_seq = ['O' for _ in seq]
        for e in input_dict['entity'].values():
            et = e['entity']
            word = e['word']
            match_res = re.finditer(re.escape(word), text)
            for res in match_res:
                if label_seq[res.start()] != 'O':
                    continue
                for i in range(res.start(), res.end()):
                    label_seq[i] = et
                break
    else:
        seq = text.split(' ')
        label_seq = ['O' for _ in seq]
        for e in input_dict['entity'].values():
            et = e['entity']
            word = e['word']
            word_lst = re.sub('\( \d+ \)', '', word).strip().split(' ')
            indices = find_sublist_indices(word_lst, seq)
            for index in indices:
                if label_seq[index] != 'O':
                    continue
                for i, _ in enumerate(word_lst):
                    label_seq[index + i] = et
                break
    return label_seq


def evaluation(predictions, labels, raw_text, zh=False):
    preds_seq, labels_seq = [], []
    for index in range(len(raw_text)):
        preds_dict = {
            'text': raw_text[index],
            'entity': {}
        }
        for e in predictions[index]:
            for k, v in e.items():
                preds_dict['entity'].update({len(list(preds_dict['entity'].values())): {'entity': v, 'word': k}})
        labels_dict = {
            'text': raw_text[index],
            'entity': labels[index]
        }
        if labels[index] == {}:
            continue
        preds_seq.extend(label_to_seq(preds_dict, zh=zh))
        labels_seq.extend(label_to_seq(labels_dict, zh=zh))
    # test
    precision, recall, f1, support = precision_recall_fscore_support(labels_seq, preds_seq,
                                                                     labels=['ORG', 'LOC', 'PER'], average='weighted',
                                                                     zero_division=0)

    micro_f = f1_score(labels_seq, preds_seq, labels=['ORG', 'LOC', 'PER'], average='micro', zero_division=0)
    macro_f = f1_score(labels_seq, preds_seq, labels=['ORG', 'LOC', 'PER'], average='macro', zero_division=0)
    return micro_f, macro_f, f1


if __name__ == '__main__':
    pass
