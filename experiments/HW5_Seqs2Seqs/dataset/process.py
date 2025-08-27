# process.py -- pre-process the data before training

import re
from pathlib import Path
from tqdm import tqdm
import random
import os
import sentencepiece as spm
import subprocess
import sys

def strQ2B(ustring):
    # reference: https://cloud.tencent.com/developer/article/1435475
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):   # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace('-', '') # 删除 '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # 保留标点符号
    elif lang == 'zh':
        s = strQ2B(s) # 把字符串全角转半角
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # 保留标点符号
    s = ' '.join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r', encoding='utf-8') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r', encoding='utf-8') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w', encoding='utf-8') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w', encoding='utf-8') as l2_out_f:
                    for s1 in tqdm(l1_in_f):
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # 删除过短数据
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # 删除过长数据
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # 删除翻译前后数据长度比例超过一定值的字段
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

if __name__ == '__main__':
    clean_corpus('./data/raw', 'en', 'zh')

    prefix, src_lang, tgt_lang = Path('./data'), 'en', 'zh'
    data_prefix = './data/raw'
    data_dir = './data'
    train_ratio = 0.99
    if (prefix/f'train.clean.{src_lang}').exists() \
        and (prefix/f'train.clean.{tgt_lang}').exists() \
        and (prefix/f'valid.clean.{src_lang}').exists() \
        and (prefix/f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
    else:
        line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}', encoding='utf-8'))
        labels = list(range(line_num))
        random.shuffle(labels)
        for lang in [src_lang, tgt_lang]:
            train_f = open(os.path.join(data_dir, f'train.clean.{lang}'), 'w', encoding='utf-8')
            valid_f = open(os.path.join(data_dir, f'valid.clean.{lang}'), 'w', encoding='utf-8')
            count = 0
            # 基于下标拆分训练和测试集
            for line in open(f'{data_prefix}.clean.{lang}', 'r', encoding='utf-8'):
                if labels[count]/line_num < train_ratio:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
            train_f.close()
            valid_f.close()

    vocab_size = 8000
    if (prefix/f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
    else:
        spm.SentencePieceTrainer.Train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix/f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram', # 用'bpe'也行 
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )
    
    # 用训练好的模型清洗数据: 将句子的起始终点加上标识：
    # 如 ▁這個 研 討 會 給我 留 下 了 極 為 深 刻 的 印 象 ▁, ▁我想 感 謝 大家 對我 之前 演講 的 好 評 
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(model_file=str(prefix/f'spm{vocab_size}.model'))
    in_tag = {
        'train': 'train.clean',
        'valid': 'valid.clean',
        'test': 'test.raw.clean',
    }
    for split in ['train', 'valid', 'test']:
        for lang in [src_lang, tgt_lang]:
            out_path = prefix/f'{split}.{lang}'
            if out_path.exists():
                print(f"{out_path} exists. skipping spm_encode.")
            else:
                with open(prefix/f'{split}.{lang}', 'w', encoding='utf-8') as out_f:
                    with open(prefix/f'{in_tag[split]}.{lang}', 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.strip()
                            tok = spm_model.Encode(line, out_type=str)
                            print(' '.join(tok), file=out_f)

    dataset_name = 'data-bin'
    binpath = Path('./data', dataset_name)

    cmd = [
    'python', '-m', 'fairseq_cli.preprocess',
    '--source-lang', src_lang,
    '--target-lang', tgt_lang,
    '--trainpref', str(prefix / 'train'),
    '--validpref', str(prefix / 'valid'),
    '--testpref', str(prefix / 'test'),
    '--destdir', str(binpath),
    '--joined-dictionary',
    '--workers', '2']
    
    
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
        sys.exit(0)
    else:
        try:
            subprocess.run(cmd, check=True)
            print("Fairseq preprocessing completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error during preprocessing: {e}")
            sys.exit(1)