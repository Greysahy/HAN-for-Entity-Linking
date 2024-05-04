import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init


class AttentionalBiGRU(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0):
        super(AttentionalBiGRU, self).__init__()
        self.register_buffer("mask",torch.FloatTensor())

        natt = hid_size*2

        self.gru = nn.GRU(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
        self.lin = nn.Linear(hid_size*2,natt)
        self.dropout = nn.Dropout(0.2)
        self.att_w = nn.Linear(natt,1,bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, packed_batch):

        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        emb_h = self.tanh(self.dropout(self.lin(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1))))  # Nwords * Emb

        attend = self.att_w(emb_h).view(enc_sents.size(0),enc_sents.size(1)).transpose(0,1)
        all_att = self._masked_softmax(attend,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents

        return attended.sum(0,True).squeeze(0)

    def forward_att(self, packed_batch):
        
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        
        emb_h = self.tanh(self.dropout(self.lin(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1))))  # Nwords * Emb
        attend = self.att_w(emb_h).view(enc_sents.size(0),enc_sents.size(1)).transpose(0,1)
        all_att = self._masked_softmax(attend,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent 
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
        return attended.sum(0,True).squeeze(0), all_att

    def _list_to_bytemask(self,l):
        mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)

        for i,j in enumerate(l):
            if j != l[0]:
                mask[i,j:l[0]] = 0

        return mask

    def _masked_softmax(self,mat,mask):
        exp = torch.exp(mat) * Variable(mask,requires_grad=False)
        sum_exp = exp.sum(1,True)+0.0001
     
        return exp/sum_exp.expand_as(exp)


class HierarchicalDoc(nn.Module):

    def __init__(self, ntoken, emb_size=300, hid_size=150):
        super(HierarchicalDoc, self).__init__()

        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.word = AttentionalBiGRU(emb_size, hid_size)
        self.sent = AttentionalBiGRU(hid_size*2, hid_size)
        self.score = nn.Linear(2, 1)
        self.emb_size = emb_size
        self.register_buffer("documents",torch.Tensor())
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.size()) > 1:  # For weights, not biases
                    init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

    def set_emb_tensor(self,emb_tensor):
        self.embed.weight.data = emb_tensor

    def _reorder_sent(self,sents,stats):
        
        sort_r = sorted([(l,r,s,i) for i,(l,r,s) in enumerate(stats)], key=itemgetter(0,1,2)) #(len(r),r#,s#)
        builder = OrderedDict()
        
        for (l,r,s,i) in sort_r:
            if r not in builder:
                builder[r] = [i]
            else:
                builder[r].append(i)
                
        list_r = list(reversed(builder))
        
        revs = Variable(self._buffers["documents"].resize_(len(builder),len(builder[list_r[0]]),sents.size(1)).fill_(0), requires_grad=False)
        lens = []
        real_order = []
        
        for i,x in enumerate(list_r):
            revs[i,0:len(builder[x]),:] = sents[builder[x],:]
            lens.append(len(builder[x]))
            real_order.append(x)

        real_order = sorted(range(len(real_order)), key=lambda k: real_order[k])
        
        return revs,lens,real_order

    def cos_sim(self, v, candidates):
        # 由于每个文档句子个数不同，直觉上分batch独立计算余弦相似度
        # 考虑过pad后做并行运算，但由于句子的个数不同，可能需要pad到lcm的规模
        # 而数据量本身不大，因此就没再深入考虑
        sims = []
        for idx, v_i in enumerate(v):
            candidate_t = candidates[idx]
            v_i = v_i.unsqueeze(0).expand(candidate_t.shape[0], -1)
            sim = F.cosine_similarity(v_i, candidate_t, dim=1)
            sims.append(sim)
        return sims

    def forward(self, batch_documents,stats, sent_idx, candidates, targets, sent_nums):
        """
        batch_documents: 长度为句子个数，存储了所有batch的句子向量
        stats：句子长度，文档的句子个数，batch_size, 句子索引
        sent_idx：长度为batch_size，存储了每个实体对应的所在文档的句子号
        candidates： 长度为batch_size，存储了每个batch的候选实体矩阵
        targets：长度为batch_size：存储每个batch对应的目标实体索引
        sent_nums：长度为batch_size，存储每个batch的句子个数
        """
        ls,lr,rn,sn = zip(*stats) # (sentence_length, doc_length, batch_idx, sentence_idx)
        emb_w = F.dropout(self.embed(batch_documents),training=self.training) # (句子个数，句子长度，嵌入维度=300)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
        sent_embs = self.word(packed_sents) #(句子个数，嵌入维度)
        
        rev_embs,lens,real_order = self._reorder_sent(sent_embs,zip(lr,rn,sn))
        
        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lens, batch_first=True)
        doc_embs = self.sent(packed_rev)
        # 防止分句出错, 句子index溢出
        sent_idx = [min(idx, len(sent_embs) - 1) for idx in sent_idx]

        # 取得句子表征向量
        v_dl = sent_embs[torch.tensor(sent_idx)]  # (batch_size, emb_size)
        # 取得文档表征向量
        v_d = doc_embs[real_order,:] # (batch_size, emb_size)

        # 计算表征向量和每个实体向量的余弦相似度
        sim_1 = self.cos_sim(v_dl, candidates) # (sentence_length.sum(), 1)
        sim_2 = self.cos_sim(v_d, candidates) # (sentence_length.sum(), 1)

        out = []
        label = []
        predict_right = 0
        for i, sim in enumerate(sim_1):
            x = torch.cat((sim_1[i].unsqueeze(-1), sim_2[i].unsqueeze(-1)), dim=1)  # (句子个数，2)
            # 过一个Linear(2， 1)得到维度为(句子个数，1)的分数
            scores = self.score(x).squeeze(-1)

            # 如果目标实体的分数是最高的，则预测正确
            if torch.max(scores) == scores[targets[i]]:
                predict_right += 1

            # 取topk的分数，并将目标实体的个数复制k遍,优化目标是 目标实体的分数 > topk分数
            k = min(1, scores.shape[0])
            out.append(scores[targets[i]].unsqueeze(0).repeat(k))
            top_scores, _ = torch.topk(scores, k, dim=0)
            label.append(top_scores)

        return torch.cat(out), torch.cat(label), predict_right




