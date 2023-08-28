# build transformer on data
import pandas as pd
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

from UnsupervisedLoss import my_unsupervised_loss as MyLoss


class MultiLabelModel(nn.Module):
    def __init__(self, basemodel_output, num_classes, basemodel=None):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes

        # config
        self.cfg_normalize = False  # unchecked other method, diff with embedding.
        self.cfg_has_embedding = True
        self.cfg_num_features = basemodel_output  # is there a better number?
        self.cfg_dropout_ratio = 0.  # 0. is better than 0.8 at attributes:pants problem

        # diy head
        for index, num_class in enumerate(num_classes):
            if self.cfg_has_embedding:
                setattr(self, "EmbeddingFeature_FCLayer_" + str(index),
                        nn.Linear(basemodel_output, self.cfg_num_features))
                setattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index), nn.BatchNorm1d(self.cfg_num_features))
                feat = getattr(self, "EmbeddingFeature_FCLayer_" + str(index))
                feat_bn = getattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index))
                init.kaiming_normal_(feat.weight, mode='fan_out')
                init.constant_(feat.bias, 0)
                init.constant_(feat_bn.weight, 1)
                init.constant_(feat_bn.bias, 0)
            if self.cfg_dropout_ratio > 0:
                setattr(self, "Dropout_" + str(index), nn.Dropout(self.cfg_dropout_ratio))
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(self.cfg_num_features, num_class))
            classifier = getattr(self, "FullyConnectedLayer_" + str(index))
            init.normal_(classifier.weight, std=0.001)
            init.constant_(classifier.bias, 0)

    def forward(self, x):
        if self.basemodel is not None:
            x = self.basemodel.forward(x)
        outs = list()
        for index, num_class in enumerate(self.num_classes):
            if self.cfg_has_embedding:
                feat = getattr(self, "EmbeddingFeature_FCLayer_" + str(index))
                feat_bn = getattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index))
                x = feat(x)
                x = feat_bn(x)
            if self.cfg_normalize:
                x = F.normalize(x)  # getattr bug
            elif self.cfg_has_embedding:
                x = F.relu(x)
            if self.cfg_dropout_ratio > 0:
                dropout = getattr(self, "Dropout_" + str(index))
                x = dropout(x)
            classifier = getattr(self, "FullyConnectedLayer_" + str(index))
            out = classifier(x)
            outs.append(out)
        return outs


def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k: v for k, v in pretrained_state_dict.iteritems() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict


def BuildMultiLabelModel(basemodel_output, num_classes, basemodel=None):
    return MultiLabelModel(basemodel_output, num_classes, basemodel=basemodel)


import torch
from transformers import BertTokenizer, BertModel, BertConfig
import logging
import random

import paddle
from paddlenlp.data import Stack, Tuple, Pad
import re

df = pd.read_csv("data/report_clean5.csv")
# df = df.drop(columns=['无法分类'],axis=1)
# df = df[df['embedding'].notna()]
# df['embedding'] = df['embedding'].apply(eval).apply(np.array)
# data = copy.deepcopy(df[['TradingDate', 'TradingTime','tag_trans','embedding','InnerCode','SecuCode']])
#
Tag_list = ['异常波动', '分红', '股东大会决议', '业绩预告', '持股变动', '资产重组', '再融资', '股权激励', '关联交易',
            '担保', '退市风险', '交易所交易公开信息', '现金管理', '会计政策变更', '人员聘请', '审计保留意见', 'IPO',
            '变更信息', '内部控制', '新项目开展', '诉讼案件', '承诺澄清']
for x in ['异常波动', '担保', '退市风险', 'IPO', '新项目开展', '承诺澄清', '变更信息', '交易所交易公开信息']:
    Tag_list.remove(x)


class LongTextDataloader(object):

    def __init__(self, text_array, max_sub_sentence_len, batch_size,
                 shuffle=False):
        """
        长文本dataloader，初始化函数。

        Args:
            text_list: 数据集
            max_sub_sentence_len (int): 每个子句最大的长度限制
            batch_size (int): 一次返回多少句子
            shuffle (bool): 是否打乱数据集
        """
        self.texts = self.__read_data(text_array)
        # assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'
        self.start = 0
        self.end = len(self.texts)
        self.batch_size = batch_size
        self.max_sub_sentence_len = max_sub_sentence_len
        self.visit_order = [i for i in range(self.end)]
        if shuffle:
            random.shuffle(self.visit_order)

    def __read_data(self, text_array) -> tuple:
        """
        将本地数据集读到数据加载器中。

        Args:
            filename (str): 数据集文件名

        Returns:
            [tuple] -> 文本列表，标签列表
        """
        list = []
        for line in text_array:
            line_ = re.sub(r'[0-9]+', '', line)
            try:
                pattern = re.search(r"\s[A-Za-z]{1}\s", line_).group()
                line_ = re.sub(pattern, pattern.replace(" ", ""), line_)
            except:
                line_ = line_
            # text = line_.split()
            # for i in text:
            #     if len(i)<=1:
            #         text.remove(i)
            list.append(line_)
        return list

    def __split_long_text(self, text: str) -> list:
        """
        用于迭代器返回数据样本的时候将长文本切割为若干条。

        Args:
            text (str): 长文本, e.g. -> "我爱中国"

        Returns:
            [list] -> ["我爱", "中国"]（假设self.max_sub_sentence_len = 2）
        """
        sub_texts, start, length = [], 0, len(text)
        # text_ = text.split()
        # for i in text_:
        #     if len(i)<=1:
        #         text_.remove(i)
        while start < length:
            sub_texts.append(text[start: start + self.max_sub_sentence_len])
            start += self.max_sub_sentence_len
        return sub_texts

    def __next__(self) -> dict:
        """
        迭代器，每次返回数据集中的一个样本，返回样本前会先将长文本切割为若干个短句子。

        Raises:
            StopIteration: [description]

        Returns:
            [dict] -> {
                'text': [sub_sentence 1, sub_sentence 2, ...],
                'label': 1
            }
        """
        if self.start < self.end:
            ret = self.start
            batch_end = ret + self.batch_size
            self.start += self.batch_size
            currents = self.visit_order[ret: batch_end]
            return {'text': [self.__split_long_text(self.texts[c]) for c in currents]}
        else:
            self.start = 0
            raise StopIteration

    def __iter__(self):
        return self


class BertClassifier(nn.Module):
    def __init__(self, filepath, length):
        super(BertClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(filepath + "/")
        modelConfig = BertConfig.from_pretrained(filepath + "/config.json")
        self.textExtractor = BertModel.from_pretrained(filepath + "/pytorch_model.bin", config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        # self.bert = BertModel.from_pretrained(filepath)

        # self.Model = BertModel.from_pretrained("clue/roberta_chinese_base")
        self.fc = nn.Linear(512, length)
        self.activation = nn.Tanh()

    def forward(self, sub_texts: list, max_seq_len: int, batch_size: int, max_sub_num: int):
        """
        正向传播函数，将一段长文本中的所有N段子文本都过一遍backbone，得到N个pooled_output([CLS]过了一个tanh函数)，
        再将这N个pooled_output向量Pooling成一个768-dim的融合向量，融合向量中768-dim中的每一维都取这N个向量对应dim
        的最大值（MaxPooling），使用MaxPooling而非MeanPooling是因为BERT类的模型抽取的特征非常稀疏，Max-Pooling
        会保留突出的特征，Mean-Pooling会将特征拉平。

        Args:
            sub_texts (list[str]): batch个长文本被切成的所有子段列表 -> (batch, sub_text_num, sub_text_len)
        """
        # for item in self:  # using the dataloader as an iterator

        sub_inputs = []
        for sub_text in sub_texts:  # 一个batch的句子
            sub_idx = 0
            for sub in sub_text:  # 一个句子中的子句
                if sub_idx == max_sub_num:  # 若达到最大子句数，则丢掉剩余的子句
                    break
                # sub = sub.split()
                encoded_inputs = self.tokenizer(sub, max_length=max_seq_len, padding='max_length',
                                                truncation=True)  # 添加special tokens， 也就是CLS和SEP  # pad到最大的长度)
                input_ids = torch.tensor(encoded_inputs["input_ids"])
                token_type_ids = torch.tensor(encoded_inputs["token_type_ids"])
                # input_ids, token_type_ids = paddle.to_tensor(input_ids).unsqueeze(0), paddle.to_tensor(token_type_ids).unsqueeze(0)
                # attention_mask = torch.tensor(encoded_inputs['attention_mask']).unsqueeze(0)
                sub_inputs.append([input_ids, token_type_ids])
                # print(input_ids, token_type_ids)
                sub_idx += 1
                print(sub_idx)

            while sub_idx < max_sub_num:  # 若未达到最大子句数，则用空句子填满
                encoded_inputs = self.tokenizer('', truncation=True,
                                                max_length=max_seq_len, padding='max_length')
                sub_inputs.append([torch.tensor(encoded_inputs["input_ids"]),
                                   torch.tensor(encoded_inputs["token_type_ids"])])
                sub_idx += 1
                print("aaa")
                print(sub_idx)


        sub_inputs_ = Tuple(  # (batch*max_sub_setences, seq_len)
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)  # segment
        )(sub_inputs)
        # input_ids = torch.stack(sub_inputs).squeeze()  # shape: (batch*max_sub_sentences, seq_len)
        # token_type_ids = torch.stack(sub_inputs_token).squeeze()

        # print(sub_inputs)
        input_ids, token_type_ids = sub_inputs_  # (batch*max_sub_setences, seq_len)
        input_ids, token_type_ids = torch.tensor(input_ids), torch.tensor(token_type_ids)
        print(input_ids.size())
        print('ok1')
        out = self.textExtractor(input_ids, token_type_ids)[0][:, 0, :]
        print("ok")
        # sequence_output, pooled_output = self.backbone(input_ids,token_type_ids)

        # sequence_output: (batch*max_sub_setences, seq_len, cls-dim)
        # pooled_output: (batch*max_sub_setences, cls-dim)
        # output1 = sequence_output[:, 0, :]
        # print(output1)
        print(out.size())
        pooled_output = out.reshape(-1, 1, max_sub_num, max_seq_len)
        #                                (-1, 1, 2, 768))  # (batch, 1, max_sub_setences, cls-dim)
        pooled = F.adaptive_max_pool2d(pooled_output, output_size=(1, max_seq_len)).squeeze()
        # (batch, cls-dim)
        # pooled = F.adaptive_avg_pool2d(pooled_output, output_size=(1, 768)).squeeze()     # (batch, cls-dim)
        #
        print(pooled.size())
        fc_out = self.fc(pooled)
        fc_out = self.activation(fc_out)
        print(fc_out)
        # output = self.output_layer(fc_out)  # (batch, 2)

        return fc_out


data_loader = LongTextDataloader(text_array=df.head(10).Detail,
                                 max_sub_sentence_len=248,
                                 batch_size=4,
                                 shuffle=False)
# Iterate over your text_list
embeddings = []
model = BertClassifier('./roberta', 248)
loss_fn = nn.CrossEntropyLoss()  # Or another appropriate loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epoch = 5

for epoch in range(num_epoch):
    for step, item in enumerate(data_loader, start=1):
        a = item['text']
        outputs = model(a, 248, 4, 6)
        embeddings.append(outputs)
        loss = MyLoss(inputs, outputs)
        loss.backward()
        # Update the weights
        optimizer.step()
        # Zero the gradients
        optimizer.zero_grad()
        embeddings.append(embedding)
