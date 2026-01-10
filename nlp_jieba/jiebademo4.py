# 繁体分词
# coding:utf-8
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import jieba

content = "我是小明，我愛自然語言處理"
# 精确模型：试图将句子最精确地切开，适合文本分析。也属于默认模式
result = jieba.lcut(content)

print(result)

