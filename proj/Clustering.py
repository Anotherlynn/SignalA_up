#!./.conda/envs/py310/bin/python3.10
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch

import pyLDAvis
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# 加入实体名称
import jieba

jieba.load_userdict("data/organization.txt")
jieba.load_userdict("data/organization2.txt")
jieba.load_userdict("data/person.txt")
jieba.load_userdict("data/place.txt")


# 换一个方法


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join(set([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(message)
    print()


class LDAClustering():
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def cut_words(self, sentence):
        sentence = re.sub(r'[0-9\.]+', r'', sentence)
        return ' '.join(jieba.lcut(sentence))

    def pre_process_corpus(self, corpus_path, stopwords_path):
        """
        数据预处理，将语料转换成以词频表示的向量。
        :param corpus_path: 语料路径，每条语料一行进行存放
        :param stopwords_path: 停用词路径
        :return:
        """
        data = pd.read_csv("股东大会.csv")
        # with open(corpus_path,'r',encoding='utf-8') as f:
        corpus = [self.cut_words(line.strip()) for line in data['InfoTitle']]
        stopwords = self.load_stopwords(stopwords_path)
        for i in ["股东大会","临时","股东","上海","召开","第一次","第二次","第三次"]:
            stopwords.append(i)
        self.cntVector = CountVectorizer(stop_words=stopwords)
        cntTf = self.cntVector.fit_transform(corpus)
        return cntTf

    def fmt_lda_result(self, lda_result):
        ret = {}
        for doc_index, res in enumerate(lda_result):
            li_res = list(res)
            doc_label = li_res.index(max(li_res))
            if doc_label not in ret:
                ret[doc_label] = [doc_index]
            else:
                ret[doc_label].append(doc_index)
        return ret

    def lda(self, corpus_path, n_components=5, learning_method='batch',
            max_iter=10, stopwords_path='../data/stop_words.txt'):
        """
        LDA主题模型
        :param corpus_path: 语料路径
        :param n_topics: 主题数目
        :param learning_method: 学习方法: "batch|online"
        :param max_iter: EM算法迭代次数
        :param stopwords_path: 停用词路径
        :return:
        """
        cntTf = self.pre_process_corpus(corpus_path=corpus_path, stopwords_path=stopwords_path)
        tf_feature_names = self.cntVector.get_feature_names_out()
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, learning_method=learning_method)
        docres = lda.fit_transform(cntTf)
        print('Model saved successfully!')
        print_top_words(lda, tf_feature_names, n_top_words=10)

        return self.fmt_lda_result(docres)


LDA = LDAClustering()
print("Start to build model.")
ret = LDA.lda("ata/corpus.txt", stopwords_path='data/stopwords.txt', max_iter=100, n_components=10)

print(ret)

import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# 输入文件
glove_file = 'GloVe/vectors.txt'
# 输出文件
w2v_file = 'GloVe/w2v.txt'
# 开始转换
glove2word2vec(glove_file, w2v_file)
# 加载转化后的文件
# model = KeyedVectors.load_word2vec_format(w2v_file, binary=False,encoding='utf-8')   #该加载的文件格式需要转换为utf-8
glove_wiki = KeyedVectors.load_word2vec_format(glove_file, binary=False, encoding='utf-8', no_header=True)

# 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
glove_wiki.init_sims(replace=True)
glove_wiki.save("Glove/w2v.model")
glove_wiki.save(w2v_file.replace(".txt", ".bin"))
embed_path = "GloVe/w2v.bin"
glove_wiki = gensim.models.KeyedVectors.load(embed_path, mmap='r')

weight_numpy = np.load(file="/share_v3/fangcheng/dev/instruction_intent_parse/intent_classify/intent_functional/data/emebed.ckpt.npy")
embedding =torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
word2idx = pd.read_pickle("/share_v3/fangcheng/dev/instruction_intent_parse/intent_classify/intent_functional/data/word2idx.ckpt")
idx2word = pd.read_pickle("/share_v3/fangcheng/dev/instruction_intent_parse/intent_classify/intent_functional/data/idx2word.ckpt")
sentences = ["我","爱", "北京","天安门"]
ids = torch.LongTensor([word2idx[item] for item in sentences])
wordvector = embedding(ids)
print(wordvector.shape)
# 使用gensim找最相近的词



