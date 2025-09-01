import re

instruction = '''The task is to label named entities from in the given sentence and the entity should be chosen in [{}].
Please answer in the format (entity type, entity).
Here is the sentence: 
'''


def ner_data_process(path, zh=False):
    with open(path, 'r', encoding='utf-8') as f:
        result = {}
        words, labels = [], []
        for line in f.readlines():
            if line.strip():
                words.append(line.strip().split()[0])
                labels.append(line.strip().split()[1])
            else:
                text = ' '.join(words) if not zh else re.sub("'''", "'", re.sub('#', ' ', ''.join(words)))
                index = 0
                entity = {}
                while index < len(labels):
                    if labels[index].startswith('B-') or labels[index].startswith('I-'):
                        t = index + 1
                        while t < len(labels) and labels[t].startswith('I-'):
                            t += 1
                        entity[len(entity)] = {
                            'word': ' '.join(words[index:t]) if not zh else re.sub('#', ' ', ''.join(words[index:t])),
                            'entity': labels[index][2:]
                        }
                        index = t
                    else:
                        index += 1
                result[len(result)] = {
                    'text': text,
                    'entity': entity
                }
                words, labels = [], []
        entity_type = set()
        for data in result.values():
            for entity in data['entity'].values():
                entity_type.add(entity['entity'])
        for i in result.keys():
            result[i]['instruction'] = instruction.format(', '.join(list(entity_type)))
        return result


def entity_process(entity):
    label_format = '({0}, {1})'
    return ' '.join([label_format.format(e['entity'], e['word']) for e in entity.values()])


def out_filter(text):
    pattern = r'\) '
    text = re.sub(pattern, '\n', text)
    if text.endswith(')'):
        text = text[:-1]
    result = []
    for item in text.split('\n'):
        try:
            if item.count(', ') > 1:
                item = item[1:]
                result.append({
                    ' '.join(item.split(', ')[1:]): item.split(', ')[0]
                })
            elif ', ' in item:
                item = item[1:]
                result.append({
                    item.split(', ')[1]: item.split(', ')[0]
                })
        except Exception as e:
            print(e)
            continue
    return result


if __name__ == "__main__":
    pass
