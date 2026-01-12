# Embedding
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn


# 实验：nn.Embedding层词向量可视化分析
# 1 对句子分词 word_list
# 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3 创建nn.Embedding层，查看每个token的词向量数据
# 4 创建SummaryWriter对象, 可视化词向量
#   词向量矩阵embd.weight.data 和 词向量单词列表my_token_list添加到SummaryWriter对象中
#   summarywriter.add_embedding(embd.weight.data, my_token_list)
# 5 通过tensorboard观察词向量相似性
# 6 也可通过程序，从nn.Embedding层中根据idx拿词向量

def dm_nnembeding_show():
    # 1.对句子分词 word_list
    sentence1 = '我是程序员，正在开始阶段性学习'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    # 2.对所有的句子分词
    word_list = []
    for s in sentences:
        word_list.append(jieba.lcut(s))
    print('word_list--->', word_list)

    # 3.获取 word_ index,index_word
    mytokenizer = Tokenizer()
    mytokenizer.fit_on_texts(word_list)
    print('mytokenizer.word_index--->', mytokenizer.word_index)
    print('mytokenizer.index_word--->', mytokenizer.index_word)

    # 4.将文本序列转换成数字序列
    # 4.1 my_token_list
    my_token_list = mytokenizer.index_word.values()
    print('my_token_list-->', my_token_list)
    # 4.2打印文本数值化以后的句子
    sentence2id = mytokenizer.texts_to_sequences(word_list)
    print('sentence2id--->', sentence2id, len(sentence2id))
    # 5.获取样本中的所有单词
    word = mytokenizer.index_word.values()
    print('word--->', word)

    #
    # 6.实例化Embedding对象
    # 6.1 num_embeddings：代表需要进行词向量表示的单词总个数（一定是去重）
    # 6.2 embedding_dim：代表每个单词进行词嵌入的维度，
    embd = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    print("embd--->", embd)
    # 6.3 获取词向量矩阵
    print('nn.Embedding层词向量矩阵-->', embd.weight.data, embd.weight.data.shape)

    # 7 可视化 embdding
    # 7.1 创建SummaryWriter对象 词向量矩阵embd.weight.data 和 词向量单词列表my_token_list
    summarywriter = SummaryWriter()
    summarywriter.add_embedding(embd.weight.data, my_token_list)
    summarywriter.close()

    # 7.2 通过tensorboard观察词向量相似性
    # cd 程序的当前目录下执行下面的命令
    # 启动tensorboard服务 tensorboard --logdir=runs --host 0.0.0.0
    # 通过浏览器，查看词向量可视化效果 http://127.0.0.1:6006
    # 8. 从nn.Embedding层中根据idx拿词向量
    print('从nn.Embedding层中根据idx拿词向量')
    # # 6 从nn.Embedding层中根据idx拿词向量
    for idx in range(len(mytokenizer.index_word)):
        tmpvec = embd(torch.tensor(idx))
        print('%4s' % (mytokenizer.index_word[idx + 1]), tmpvec.detach().numpy())
        print("*"*80)


if __name__ == '__main__':
    dm_nnembeding_show()
