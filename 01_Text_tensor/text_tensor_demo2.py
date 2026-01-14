# 词向量的训练保存加载
# 导入fasttext
import fasttext

# 消除FastText的警告信息（可选，仅用于整洁输出）
fasttext.FastText.eprint = lambda x: None


def dm_fasttext_train_save_load():
    # 1 使用train_unsupervised(无监督训练方法) 训练词向量
    mymodel = fasttext.train_unsupervised('./data/fil9')
    print('训练词向量 ok')

    # 2 save_model()保存已经训练好词向量
    # 注意，该行代码执行耗时很长
    mymodel.save_model("./data/fil9.bin")
    print('保存词向量 ok')

    # 3 模型加载
    mymodel = fasttext.load_model('./data/fil9.bin')
    print('加载词向量 ok')


# 通过get_word_vector方法来获得指定词汇的词向量, 默认词向量训练出来是1个单词100特征
def dm_fasttext_get_word_vector():
    mymodel = fasttext.load_model('./data/fil9.bin')

    myvector = mymodel.get_word_vector('the')
    print('myvector->', type(myvector), myvector.shape, myvector)


# 检查单词向量质量的一种简单方法就是查看其邻近单词,
# 通过我们主观来判断这些邻近单词是否与目标单词相关来粗略评定模型效果好坏.
def dm_fasttext_get_nearest_words():
    mymodel = fasttext.load_model('./data/fil9.bin')
    # 关键修正：使用get_nearest_neighbors()替代废弃的get_nearest_words()
    nearest_words = mymodel.get_nearest_neighbors('cow')
    print('mymodel.get_nearest_neighbors->', nearest_words)


# 在训练词向量过程中, 我们可以设定很多常用超参数来调节我们的模型效果, 如:
# 无监督训练模式: 'skipgram' 或者 'cbow', 默认为'skipgram',
# 在实践中，skipgram模式在利用子词方面比cbow更好.
# 词嵌入维度dim: 默认为100, 但随着语料库的增大, 词嵌入的维度往往也要更大.
# 数据循环次数epoch: 默认为5, 但当你的数据集足够大, 可能不需要那么多次.
# 学习率lr: 默认为0.05, 根据经验, 建议选择[0.01，1]范围内.
# 使用的线程数thread: 默认为12个线程, 一般建议和你的cpu核数相同.
def dm_fasttext_train_params():
    model = fasttext.train_unsupervised('./data/fil9', "cbow", dim=300, epoch=1, lr=0.1, thread=8)
    print('训练词向量 ok')

    # 2 save_model()保存已经训练好词向量
    # 注意，该行代码执行耗时很长
    model.save_model("./data/fil9_cbow.bin")
    print('保存词向量 ok')

if __name__ == '__main__':
    # dm_fasttext_train_save_load()
    # dm_fasttext_get_word_vector()
    # dm_fasttext_get_nearest_words()
    dm_fasttext_train_params()
