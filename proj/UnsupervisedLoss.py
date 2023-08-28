import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


# 1.embedding similarity: 计算嵌入向量的相似性
# 1.1 HingeEmbedding:https://blog.csdn.net/ltochange/article/details/118071383
# 1.2 CosineEmbedding
# 1.3 ContrastiveLoss对比损失
# 1.4 torch.cosine_similarity 慢
# 1.5 LSH:Locality Sensitive Hashing, 在最相似的几个样本中搜索。把原向量Hash到新的空间，在原本空间中相似的向量，在新空间中有更大概率相似：https://zhuanlan.zhihu.com/p/581008101
# sklearn老版有：https://blog.csdn.net/qysh123/article/details/113754991, https://juejin.cn/post/6886345319093633031
# 实现：https://santhoshhari.github.io/Locality-Sensitive-Hashing/


class ULoss(_Loss):
    def __init__(self, margin=0.2, reduction='mean'):
        super(ULoss, self).__init__()
        self.margin = margin
        self.reduction = reduction


    def get_label(self,emb):
        if type(emb)==np.ndarray:
            label = torch.ones(emb.shape[0])
        elif type(emb)==torch.Tensor:
            try:
                label = torch.ones(emb.size(0))
            except:
                label = torch.ones(1)
        else:
            print("Error in input type")
        return label

    def cos_loss(self, emb1, emb2):
        cosine_loss = nn.CosineEmbeddingLoss(self.margin, reduction=self.reduction)
        loss_flag = self.get_label(emb1)
        return cosine_loss(torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0), loss_flag)

    def hinge_loss(self, emb1, emb2):
        hinge_loss_fn = nn.HingeEmbeddingLoss(self.margin, reduction=self.reduction)
        x = self.cos_loss(emb1, emb2)
        y = self.get_label(x)
        return hinge_loss_fn(x, y)

    def contrastive_loss(self,emb1, emb2):
        label = self.get_label(emb1)
        distance = F.pairwise_distance(torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0))
        loss = 0.5 * (label * torch.pow(distance, 2) +
                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss.mean()



