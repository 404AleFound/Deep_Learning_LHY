# preview.py -- try reading the data for training

with open('./data/raw/raw.en', encoding='utf-8') as file:
    lines_raw_en = file.readlines()

with open('./data/raw/raw.zh', encoding='utf-8') as file:
    lines_raw_zh = file.readlines()

for i in range(5):
    print(lines_raw_en[i],'\n', lines_raw_zh[i])

print('the length of en text is: ', len(lines_raw_en))