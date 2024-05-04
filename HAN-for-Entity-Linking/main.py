import argparse
import random
import time
from datetime import timedelta

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from Nets import HierarchicalDoc
from Data import DocumentDataset, Vectorizer
import sys
import json


writer = SummaryWriter("./logs")


def setup_seed(seed):
    """
    固定随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner

def txt_logger(info):
    """
    将 info 写入训练日志txt文件
    """
    with open("log.txt", mode="a", encoding="utf-8") as f:
        f.write(info)


def check_memory(emb_size,max_sents,max_words,b_size,cuda):
    try:
        e_size = (2,b_size,max_sents,max_words,emb_size) 
        d_size = (b_size,max_sents,max_words)
        t = torch.rand(*e_size)
        db = torch.rand(*d_size)

        if cuda:
            db = db.cuda()
            t = t.cuda()

        print("-> Quick memory check : OK\n")

    except Exception as e:
        print(e)
        print("Not enough memory to handle current settings {} ".format(e_size))
        print("Try lowering sentence size and length.")
        sys.exit()
    

def load_embeddings():
    # 读取词向量
    word_emb = json.load(open("./dataset/word_embeddings.json", "rb"))

    embedding_dim = len(list(word_emb.values())[0])

    # 找到最大的索引
    max_index = max(map(int, word_emb.keys()))
    # 创建一个足够大的张量
    embedding_tensor = torch.zeros(max_index + 1, embedding_dim)

    for idx, vector in word_emb.items():
        embedding_tensor[int(idx)] = torch.tensor(vector)

    print("Loaded embedding tensor with shape:", embedding_tensor.shape)

    # 返回嵌入tensor和嵌入tensor的个数， 用于初始化embedding层的权重
    return embedding_tensor, max_index + 1


def tuple_batcher_builder(vectorizer, mode, trim=True):

    def tuple_batch(l):
        # 将l中的数据打包成独立的元组
        document, id, sent_idx, candidates, targets = zip(*l)
        # 向量化文档
        list_doc, sent_nums = vectorizer.vectorize_batch(document, trim, mode)

        # 计算 sentence_length, doc_length, batch_idx, sentence_idx
        sentence_info = []

        for batch_idx, batch_doc in enumerate(list_doc):
            for sentence_idx, sentence in enumerate(batch_doc):
                sentence_length = len(sentence)
                doc_length = len(batch_doc)
                sentence_info.append((sentence_length, doc_length, batch_idx, sentence_idx, sentence))

        stat = sorted(sentence_info, key=lambda x: x[0], reverse=True)

        max_len = stat[0][0]
        batch_t = torch.zeros(len(stat),max_len).long()

        for i,st in enumerate(stat):
            for j,sentence in enumerate(st[-1]): # s[-1] is sentence in stat tuple
                batch_t[i,j] = sentence

        stat = [(sentence_length, doc_length, batch_idx, sentence_idx) for sentence_length, doc_length, batch_idx, sentence_idx, _ in stat]

        return batch_t, stat, sent_idx, candidates, targets, sent_nums, id
    return tuple_batch


def train(epoch,net,optimizer,dataset,criterion):
    epoch_loss = 0
    ok_all = 0
    all = 0
    net.train()

    with tqdm(total=len(dataset),desc="Training") as pbar:
        for iteration, (batch_t, stat, sent_idx, candidates, targets, sent_nums, id) in enumerate(dataset):
            optimizer.zero_grad()
            out, label, r = net(batch_t, stat, sent_idx, candidates, targets, sent_nums)
            Y = torch.ones(out.shape[0])
            loss = criterion(out, label, Y)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

            ok_all += r
            all += out.shape[0]

            pbar.update(1)
            pbar.set_postfix({"acc":100 * ok_all/ all,"MRL":epoch_loss/(iteration+1)})

    Avg_loss = epoch_loss / len(dataset)
    Acc = ok_all * 100 / all
    writer.add_scalar('/accuracy/train', Acc, epoch)
    writer.add_scalar('/loss/train', Avg_loss, epoch)
    print("===> Epoch {} Training Complete: Avg. Loss: {:.4f}, {:.4f}% accuracy\n".format(epoch, Avg_loss, Acc))
    txt_logger("===> Epoch {} Training Complete: Avg. Loss: {:.4f}, {:.4f}% accuracy\n".format(epoch, Avg_loss, Acc))
    return Avg_loss, Acc


def test(epoch,net,dataset,criterion):
    epoch_loss = 0
    ok_all = 0
    all = 0
    skipped = 0

    net.eval()
    with tqdm(total=len(dataset),desc="Evaluating") as pbar:
        for iteration, (batch_t, stat, sent_idx, candidates, targets, sent_nums, id) in enumerate(dataset):
            out, label, r = net(batch_t, stat, sent_idx, candidates, targets, sent_nums)

            Y = torch.ones(out.shape[0])
            loss = criterion(out, label, Y)
            epoch_loss += loss.item()

            ok_all += r
            all += out.shape[0]

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/all, "skipped":skipped})

    Avg_loss = epoch_loss / len(dataset)
    Acc = ok_all * 100 / all
    writer.add_scalar('/accuracy/test', Acc, epoch)
    writer.add_scalar('/loss/test', Avg_loss, epoch)

    print(f"===> TEST Complete:Avg. Loss: {Avg_loss}  {Acc}% accuracy\n\n")
    txt_logger(f"===> TEST Complete:Avg. Loss: {Avg_loss}  {Acc}% accuracy\n\n")
    return Avg_loss, Acc


def main(args):
    print(50*"-"+"\nHierarchical Attention Network for Entity Linking:\n" + 50*"-")

    print(18*"-" + "\nLoading Data:\n" + 18*"-")

    setup_seed(42)

    train_set, test_set = DocumentDataset.build_train_test()
    vectorizer = Vectorizer(max_word_len=64, max_sent_len=32);
    
    # collate_function
    train_batch = tuple_batcher_builder(vectorizer, mode="train", trim=True)
    test_batch = tuple_batcher_builder(vectorizer, mode="test", trim=True)
    
    # 初始化 dataloader
    train_loader = DataLoader(train_set, batch_size=args.b_size, shuffle=False, num_workers=0, collate_fn=train_batch, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.b_size, shuffle=False, num_workers=0, collate_fn=test_batch)

    print("Train set length:",len(train_set))
    print("Test set length:",len(test_set))

    word_emb, ntoken = load_embeddings()
    net = HierarchicalDoc(ntoken=ntoken,emb_size=args.emb_size, hid_size=args.hid_size)
    net.set_emb_tensor(word_emb)

    criterion = torch.nn.MarginRankingLoss()

    if args.cuda:
        net.cuda()
        
    print("-"*20)

    check_memory(args.max_sents,args.max_words,net.emb_size,args.b_size,args.cuda)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-2)
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)

    best_train_acc = 0
    best_test_acc = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, net, optimizer, train_loader, criterion)
        best_train_acc = max(best_train_acc, train_acc)
        test_loss, test_acc = test(epoch, net, test_loader, criterion)
        best_test_acc = max(best_test_acc, test_acc)

        print(f"Best training acc: {best_train_acc}%, test acc: {best_test_acc}% \n")

    end_time = time.time()
    txt_logger(f"Training time: {str(timedelta(seconds=(end_time - start_time)))}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Entity Linking')
    parser.add_argument("--emb-size",type=int,default=300)
    parser.add_argument("--hid-size",type=int,default=150)
    parser.add_argument("--b-size", type=int, default=4)
    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=60)
    parser.add_argument("--clip-grad", type=float,default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-words", type=int,default=64)
    parser.add_argument("--max-sents",type=int,default=32)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--emb", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--weight-classes", action='store_true')
    parser.add_argument("--output", type=str)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--balance', action='store_true',
                        help='balance class in batches')
    args = parser.parse_args()

    main(args)
