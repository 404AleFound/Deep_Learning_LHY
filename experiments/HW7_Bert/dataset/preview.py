# preview.py -- 
# Le Jiang 
# 2025/8/17

'''
训练集 31690个问答对
验证集 4131个问答对
测试集 4957个问答对

- {train/dev/test}_questions:	字典列表，包含以下键：
  - id (整数)
  - paragraph_id (整数)
  - question_text (字符)
  - answer_text (字符)
  - answer_start (整数)
  - answer_end (整数)

- {train/dev/test}_paragraphs: 
  - 字符串列表
  - 问题中的段落 ID 对应于段落列表中的索引
  - 一个段落可能被多个问题使用
'''

import json

with open('./data/ml2022spring-hw7/hw7_train.json', encoding='utf-8') as f:
    train_data = json.load(f)
    print(train_data.keys()) # keys are questions and paragraphs

questions = train_data['questions']
paragraphs = train_data['paragraphs']
question_0 = questions[0]
paragraph_0 = paragraphs[question_0['paragraph_id']]
print(question_0)
print(paragraph_0)
print(paragraph_0[question_0['answer_start']:question_0['answer_end']+1])

