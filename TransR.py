import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import Model

# @inproceedings{han2018openke,
#    title={OpenKE: An Open Toolkit for Knowledge Embedding},
#    author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
#    booktitle={Proceedings of EMNLP},
#    year={2018}
# }

class TransR(Model):
    def __init__(self, ent_num, rel_num, e_dim = 100, r_dim=100, p_norm=1, norm_flag=True, rand_init = False, margin=None):
        super(TransR, self).__init__(ent_num, rel_num)

        self.e_dim = e_dim
        self.r_dim = r_dim
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.rand_init = rand_init

        self.ent_emb = nn.Embedding(self.ent_num, self.e_dim)
        self.rel_emb = nn.Embedding(self.rel_num, self.r_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

        self.transfer_matrix = nn.Embedding(self.rel_num, self.e_dim*self.r_dim)

        if not self.rand_init:
            identity = torch.zeros(self.e_dim, self.r_dim)
            for i in range(min(self.e_dim, self.r_dim)):
                identity[i][i] = 1

            identity = identity.view(self.r_dim*self.e_dim)

            for i in range(self.rel_num):
                self.transfer_matrix.weight.data[i] = identity[i]

        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    # Calculate score function for TransR
    def calculate_score(self, head, relation, tail, mode):
        if self.norm_flag:
            head = F.normalize(head, 2, -1)
            relation = F.normalize(relation, 2, -1)
            tail = F.normalize(tail, 2, -1)

        if mode != 'normal':
            head = head.view(-1, relation.shape[0], head.shape[-1])
            tail = tail.view(-1, relation.shape[0], tail.shape[-1])
            relation = relation.view(-1, relation.shape[0], relation.shape[-1])

        if mode == 'head_batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        # Normalize the score
        score = torch.norm(score, self.p_norm, -1).flatten()

        return score

    # Project entity into relation space
    def transfer(self, ent, rel_transfer_mat):
        rel_transfer_mat = rel_transfer_mat.view(-1, self.e_dim, self.r_dim)

        if ent.shape[0] != rel_transfer_mat.shape[0]:
            ent = ent.view(-1, rel_transfer_mat.shape[0], self.e_dim).permute(1, 0, 2)
            ent = torch.matmul(e, rel_transfer_mat).permute(1, 0, 2)
        else:
            ent = ent.view(-1, 1, self.e_dim)
            ent = torch.matmul(ent, rel_transfer_mat)
        return ent.view(-1, self.r_dim)

    def forward(self, data):
        batch_head = data['batch']