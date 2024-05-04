import random
import torch
import pickle as pkl
from torch.utils.data import Dataset
import json


class DocumentDataset(Dataset):
    def __init__(self, data_path):
        super(DocumentDataset).__init__()
        # 加载训练数据
        self.docs = pkl.load(open(data_path, "rb"))["documents"]
        # 加载所有实体表征向量
        self.entity_vectors = json.load(open("./dataset/entity_vectors.json", "rb"))
        # 将表征向量转化为tensor
        for key in self.entity_vectors:
            self.entity_vectors[key] = torch.tensor(self.entity_vectors[key], dtype=torch.float32)

        # 每个mention作为一个单独的训练数据
        self.data = []
        for doc in self.docs:
            doc_text = doc['document']
            doc_id = doc['id']
            for mention in doc['mentions']:
                item = {
                    'id': doc_id,
                    'document': doc_text,
                    'mention': mention
                }
                self.data.append(item)

    def convert_candidates(self, mention):
        """
        输入： mention str
        功能： 将候选实体字符串转化为每个候选实体对应的实体向量
        """
        if torch.is_tensor(mention.get('candidates', '')):
            return
        candidates_str = mention.get('candidates', '').strip()
        parts = candidates_str.split('\t')

        # Replace candidates with embedding vectors
        candidates_tensors = []
        for i in range(0, len(parts), 2):
            entity_id = parts[i + 1]
            if entity_id in self.entity_vectors:
                tensor = self.entity_vectors[entity_id]
            else:
                random_tensor = torch.randn(300)
                tensor = random_tensor / torch.norm(random_tensor, p=2)

            candidates_tensors.append(tensor)
        mention['candidates'] = torch.stack(candidates_tensors) if candidates_tensors else torch.tensor([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        u = self.data[index]
        doc = u["document"]
        id = u["id"]
        mention = u["mention"]

        self.convert_candidates(mention)
        sent_idx = mention['sent_index']
        candidates = mention['candidates']
        target = mention['gold_index'] - 1

        return (doc, id, sent_idx, candidates, target)
    
    @staticmethod
    def build_train_test():
        return DocumentDataset("./dataset/train.pkl"), DocumentDataset("./dataset/test.pkl")


class Vectorizer():
    def __init__(self,word_dict=None,max_sent_len=8,max_word_len=32):
        # 读取停用词
        self.stop_words = set()
        with open("./dataset/stopword.txt", 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()
                self.stop_words.add(word)
        # 读取预处理的word_dict{“word” : word_id}
        self.word_dict = json.load(open("./dataset/word_dict.json", "rb"))
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len

    def vectorize_batch(self, t, mode,trim=True):
        return self._vect_dict(t, mode, trim)

    def random_append(self, s, word, p):
        if random.random() < p:
            s.append(self.word_dict[word])
        return s

    def _vect_dict(self, t, mode, trim):
        if self.word_dict is None:
            print("No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first")
            raise Exception

        # 将文档转化为若干个句子向量
        docs = []
        sent_nums = []
        for doc in t:
            document = []
            # Spacy sents sucks! 分句出错会导致计算相似度时选择了错误的句子表征向量
            # 测试后发现直接使用split效果比spacy和nltk的分句效果都好
            sents = doc.split('.')[:-1]
            for j,sent in enumerate(sents):
                # print(j, ":", sent)
                if trim and j >= self.max_sent_len:
                    break
                s = []
                words = sent.split(' ')
                for k,word in enumerate(words):
                    word = word.lower()

                    if trim and k >= self.max_word_len:
                        break

                    # 停用词或者不在word_dict中的单词直接忽略
                    if word in self.word_dict:
                        if word in self.stop_words:
                            continue
                        else:
                            s = self.random_append(s, word, p=0.6) # 仅保留60%的输入进行训练和测试
                 
                if len(s) >= 1:
                    document.append(torch.LongTensor(s))
            sent_nums.append(len(document))
            if len(document) == 0:
                document = [torch.LongTensor([0])]
            docs.append(document)

        return docs, sent_nums

