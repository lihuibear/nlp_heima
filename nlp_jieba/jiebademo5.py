# 用户自定义词典
# 添加自定义词典后, jieba能够准确识别词典中出现的词汇，提升整体的识别准确率。
# 词典格式: 每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。

# coding:utf-8
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import jieba
sentence = "我是小明，我爱自然语言处理"
# 1 没有使用用户自定义词典
mydata = jieba.lcut(sentence, cut_all=False)
print('mydata-->', mydata)

# 2 使用用户自定义词典
jieba.load_userdict("./userdict.txt")
mydata2 = jieba.lcut(sentence, cut_all=False)
print('mydata2-->', mydata2)

# mydata--> ['我', '是', '小明', '，', '我', '爱', '自然语言', '处理']
# mydata2--> ['我', '是', '小明', '，', '我', '爱', '自然语言处理']