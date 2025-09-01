def qwen14_generate(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def qwen14_multi_round_cot_label(model, tokenizer, language_name, phrase, sentence, full_output=False):
    messages = [
        {
            'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': f'You are a senior {language_name}-English translation master, please help me translate some English sentences into {language_name}. Note that the correct result should be a phrase that appears in the sentence.\nI will give you an English phrase and a {language_name} sentence, where the English phrase is the translation result in the {language_name} sentence. I want to find the raw {language_name} phrase of the English translation result.\nPlease think step by step.'
        },
        {
            'role': 'assistant',
            'content': f"Sure, I can assist with that. Please provide the English phrase and the {language_name} sentence it appears in so I can identify the corresponding {language_name} phrase for you.\n\nPlease format the input as follows:\n\nEnglish phrase: [insert English phrase]\n{language_name} sentence: [insert {language_name} sentence]\n\nI'll then provide the raw {language_name} phrase that corresponds to the English translation result."
        },
    ]
    prompts = [
        {
            'role': 'user',
            'content': f'English phrase: {phrase}\n{language_name} sentence: {sentence}'
        },
        {
            'role': 'user',
            'content': f'Check the result and make sure each word of the result appears in the given {language_name} sentence.'
        },
        {
            'role': 'user', 'content': f'Give me the final result without other words.'
        }
    ]
    for prompt in prompts:
        messages.append(prompt)
        next_round = qwen14_generate(model, tokenizer, messages)
        messages.append(
            {
                'role': 'assistant',
                'content': next_round
            }
        )
    if full_output:
        return messages
    else:
        return messages[-1]['content']


def qwen14_multi_round_cot_text(model, tokenizer, language_name, sentence, entity_types):
    if len(sentence) < 2:
        return 'None'

    messages = [
        {
            'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
            'role': 'system'
        },
        {
            'content': f'You are a senior {language_name}-English translation master, please help me translate some {language_name} sentences into English.\nNote that part of the given {language_name} sentences may contain named entities which can be labeled as [{", ".join(entity_types)}]. Try your best to recognize them and give the accurate translation of them.',
            'role': 'user'
        },
        {
            'content': f"Of course! Please provide the {language_name} sentences you would like translated, and I'll do my best to accurately translate them while recognizing any named entities following notation: [{'], ['.join(entity_types)}].",
            'role': 'assistant'
        },
    ]
    prompts = [
        {
            'content': f'The {language_name} sentence is: {sentence}\nAnalyze the given sentence and recognize the possible named entities which can be labeled as [{", ".join(entity_types)}] in the sentence.',
            'role': 'user'
        },
        {
            'content': 'Take these entities in mind and translate the sentence more accurately.',
            'role': 'user'
        },
        {
            'content': 'Give me the final result without other words.',
            'role': 'user'
        }
    ]
    for d in prompts:
        messages.append(d)
        next_round = qwen14_generate(model, tokenizer, messages)
        messages.append(
            {
                'role': 'assistant',
                'content': next_round
            }
        )
    return messages[-1]['content']


if __name__ == '__main__':
    pass
