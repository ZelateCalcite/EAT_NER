import sys
import os
import json

try:
    from eval.eval_utils import evaluation
    from utils.data_processor import ner_data_process, out_filter
except ModuleNotFoundError:
    python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print('Current path changed to: ', python_path)
    sys.path.insert(0, python_path)
    from eval.eval_utils import evaluation
    from utils.data_processor import ner_data_process, out_filter


def remove_stop_words(test_res):
    stop_word = [
        'REDIRECCIÓN',
        'WEITERLEITUNG',
        '\'', '\'\'', '``', '`', '\"', '\"\"'
    ]
    fil = []
    for i in range(len(test_res)):
        lst = []
        for j in range(len(test_res[i])):
            temp = []
            key = list(test_res[i][j].keys())[0].split(' ')
            key = filter(lambda x: x.upper() not in stop_word, key)

            if temp:
                r = []
                for index in range(len(temp)):
                    r.append(key[:temp[index]])
                    if index < len(temp) - 1:
                        r.append(key[temp[index] + 1:temp[index + 1]])
                    else:
                        r.append(key[temp[index] + 1:])
                r = filter(lambda x: x != [], r)
                for t in r:
                    lst.append({
                        ' '.join(t): list(test_res[i][j].values())[0]
                    })
            else:
                lst.append({
                    ' '.join(key): list(test_res[i][j].values())[0]
                })
        fil.append(lst)
    return fil


def print_test(test_set_path, test_tgt_path, zh=False):
    test_seq = list(ner_data_process(test_set_path, zh).values())[:]
    test_res = json.load(open(test_tgt_path, 'r', encoding='utf-8'))

    if isinstance(test_res, dict):
        test_res = test_res['result']

    fil = remove_stop_words(test_res)
    res = evaluation(
        fil,
        [i['entity'] for i in test_seq],
        [i['text'] for i in test_seq],
        zh
    )
    print(res)


if __name__ == '__main__':
    pass
