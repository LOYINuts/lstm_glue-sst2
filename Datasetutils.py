import os
from io import open
import torch
import torch.utils.data.dataset as Dataset

labels = {"positive": 1, "negative": 0}
dataset_types = {"train": "train.txt", "test": "test.txt", "valid": "valid.txt"}



class Dictionary(object):
    """
    字典,包含word2idx和idx2word
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

# 一个全局字典
ALL_DICT = Dictionary()

class Mydataset(Dataset.Dataset):
    def __init__(self, path:str, seq_len:int, dataset_type: str):
        # 最大序列长度，不够则用<pad>填充
        self.seq_len = seq_len
        # 将文本token化
        self.Data,self.Label = self.tokenize(os.path.join(path, dataset_types[dataset_type]))

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

    def tokenize(self, path):
        assert os.path.exists(path)
        # 将单词加入词典
        ALL_DICT.add_word("<pad>")  # 用于padding
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + [
                    "<eos>"
                ]  
                for word in words:
                    ALL_DICT.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            data = []
            label = []
            # sst2按行构建输入, 长于seq_len的句子进行截断，短于seq_len的用<pad>补齐长度至seq_len
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for i, word in enumerate(words):
                    if i < self.seq_len:
                        ids.append(ALL_DICT.word2idx[word])
                    else:
                        # 超过长度直接截断
                        break
                # 短于长度进行补齐
                while len(ids) < self.seq_len:
                    ids.append(ALL_DICT.word2idx["<pad>"])
                data.append(torch.tensor(ids).type(torch.int64))
                label.append(labels[words[-2]])
            
            data = torch.stack(data)
            label = torch.tensor(label)
            return data,label
