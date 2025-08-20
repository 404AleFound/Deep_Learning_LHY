# dataset.py --
# Le Jiang
# 2025/8/17

import torch 
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import json
import doctest

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

def read_data(file):# ->(list,list)
    '''
    ---
    TEST
    ---
    >>> questions, token_questions, token_paragraphs = read_data('./data/ml2022spring-hw7/hw7_train.json')
    >>> print(questions[0])
    {'id': 0, 'paragraph_id': 3884, 'question_text': '羅馬教皇利奧三世在800年正式加冕誰為羅馬人的皇帝?', 'answer_text': '查理大帝', 'answer_start': 141, 'answer_end': 144}
    >>> print(token_questions[0].ids)
    [5397, 7679, 3136, 4640, 1164, 1953, 676, 686, 1762, 8280, 2399, 3633, 2466, 1217, 1089, 6306, 4158, 5397, 7679, 782, 4638, 4640, 2370, 136]
    >>> print(len(token_paragraphs[questions[0]['paragraph_id']].ids))
    680
    '''
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    questions = data['questions']
    token_questions = tokenizer([question['question_text'] for question in questions], add_special_tokens=False)
    token_paragraphs = tokenizer([paragraph for paragraph in data['paragraphs']], add_special_tokens = False)
    # questions and paragraphs are two lists containing the text of quest* and para*

    return questions, token_questions, token_paragraphs



class My_Dataset(Dataset):
    '''
    ---
    TEST
    ---
    >>> my_dataset_train = My_Dataset('./data/ml2022spring-hw7/hw7_train.json', 'train')
    >>> my_dataset_test = My_Dataset('./data/ml2022spring-hw7/hw7_test.json', 'test')
    >>> input, input_type, input_mask, start, end = my_dataset_train[0]
    >>> print(len(input), input[0:5], input[-6:-1])
    193 tensor([ 101, 5397, 7679, 3136, 4640]) tensor([0, 0, 0, 0, 0])
    >>> print(len(input_type), input_type[0:5], input_type[-6:-1])
    193 tensor([0, 0, 0, 0, 0]) tensor([0, 0, 0, 0, 0])
    >>> print(len(input_mask), input_mask[0:5], input_mask[-6:-1])
    193 tensor([1, 1, 1, 1, 1]) tensor([0, 0, 0, 0, 0])
    >>> print(start)
    100
    >>> print(end)
    103
    >>> print(len(my_dataset_train))
    31690
    >>> input_list, input_type_list, input_mask_list = my_dataset_test[0]
    >>> print(input_list.shape, input_type_list.shape, input_mask_list.shape)
    torch.Size([4, 193]) torch.Size([4, 193]) torch.Size([4, 193])
    '''
    def __init__(self, filename, tag='train'):
        self.tag = tag
        self.questions, self.token_questions, self.token_paragraphs = read_data(filename)
        self.max_question_len = 40
        self.max_paragraph_len = 150
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1
        self.doc_stride = 150
    
    def __getitem__(self, index):
        question = self.questions[index]
        token_question = self.token_questions[index]
        token_paragraph = self.token_paragraphs[question['paragraph_id']]
        if self.tag == 'train':
            # find the answer position in tokenized-paragraph
            answer_start_token = token_paragraph.char_to_token(question["answer_start"])
            answer_end_token = token_paragraph.char_to_token(question["answer_end"])
            # create a window for paragraph
            window_mid = (answer_start_token + answer_end_token) // 2
            window_start = max(0, min(window_mid - self.max_paragraph_len // 2, len(token_paragraph) - self.max_paragraph_len))
            window_end = window_start + self.max_paragraph_len
            # re-construct tokenized-question and tokenized-paragraph
            input_question = [101] + token_question.ids[:self.max_question_len] + [102] 
            input_paragraph = token_paragraph.ids[window_start:window_end] + [102]
            # concate and then return 
            input_ids, token_type_ids, attention_mask = self.padding(input_question, input_paragraph)
            # re_locate the answer in the new sequence 	
            answer_start_token = answer_start_token - window_start + len(input_question)
            answer_end_token = answer_end_token - window_start + len(input_question)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token
        
        elif self.tag == 'eval':
            answer_txt = question['answer_text']
            input_list, input_type_list, input_mask_list = [], [], []
            for i in range(0, len(token_paragraph), self.doc_stride):
                input_question = [101] + token_question.ids[:self.max_question_len] + [102] 
                input_paragraph = token_question.ids[i : i + self.doc_stride] + [102]
                input, input_type, input_mask = self.padding(input_question, input_paragraph)
                input_list.append(input)
                input_type_list.append(input_type)
                input_mask_list.append(input_mask)
            return torch.tensor(input_list), torch.tensor(input_type_list), torch.tensor(input_mask_list), answer_txt
        
        elif self.tag == 'test':
            input_list, input_type_list, input_mask_list = [], [], []
            for i in range(0, len(token_paragraph), self.doc_stride):
                input_question = [101] + token_question.ids[:self.max_question_len] + [102] 
                input_paragraph = token_question.ids[i : i + self.doc_stride] + [102]
                input, input_type, input_mask = self.padding(input_question, input_paragraph)
                input_list.append(input)
                input_type_list.append(input_type)
                input_mask_list.append(input_mask)
            return torch.tensor(input_list), torch.tensor(input_type_list), torch.tensor(input_mask_list)
    
    def __len__(self):
        return len(self.questions)
    
    def padding(self, input_question, input_paragraph):
        padding_len = self.max_seq_len - len(input_question) - len(input_paragraph)
        input = input_question + input_paragraph + [0] * padding_len
        input_type = [0] * len(input_question) + [1] * (len(input_paragraph)) + [0] * padding_len
        input_mask = [1] * (len(input_question) + len(input_paragraph)) + [0] * padding_len
        return input, input_type, input_mask
    
if __name__ == '__main__':
    doctest.testmod(verbose=True)
