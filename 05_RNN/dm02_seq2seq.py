# -*-coding:utf-8-*-
# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 指定设备windows：是否在GPU或者CPU进行训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 指定设备M1:
# device = torch.device('mps')
# device = torch.device('cpu')
# print(device)
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = './data/eng-fra-v2.txt'


# todo:1 定义字符串的清洗函数
def norm_string(s):
    # s代表输入的字符串
    s1 = s.lower().strip()
    # 在.!?标点符号前加上空格
    s2 = re.sub(r'([.?!])', r' \1', s1)
    # 将字符串中所有的除了大小写字母以及.!?标点符号的其他字符都用空格替换
    s3 = re.sub('[^a-zA-Z.?!]+', r' ', s2)
    return s3


# todo:2 读取文件，获取样本并且获得英文词典以及法文词典

def get_data():
    # 2.1 读取文档数据
    with open('data/eng-fra-v2.txt') as fr:
        sequences = fr.read().strip().split('\n')
    # print(f'sequences--》{len(sequences)}')
    # print(f'sequences[:2]-->{sequences[:2]}')

    # 2.2 获取pair对['i m .\tj ai ans .']-->[[i m ., j ai ans .]]-->[[英文句子,法文句子],[英文句子,法文句子],....]
    eng_fra_pairs = [[norm_string(s) for s in line.split('\t')] for line in sequences]
    # print(f'eng_fra_pairs[:2]-->{eng_fra_pairs[:2]}')

    # 2.3 遍历上述的pair对，获得英文字典以及法文字典
    # 2.3.1 获取word2index
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2

    # 开始遍历
    for pair in eng_fra_pairs:
        # 构建英文词典
        # print(f'pair--》{pair}')
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        # 构建法文字典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 2.3.2 获取index2word
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, eng_fra_pairs


english_word2index, english_index2word, english_word_n, french_word2index, \
    french_index2word, french_word_n, eng_fra_pairs = get_data()


# todo:3 构建dataset数据源


class SeqDataset(Dataset):
    def __init__(self, eng_fre_paris):
        super().__init__()
        # 获取样本对
        self.pairs = eng_fre_paris
        # 获取样本的总量
        self.sample_len = len(eng_fre_paris)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 异常值修正
        item = min(max(item, 0), self.sample_len - 1)
        # 根据索引取出样本
        # 取出英文句子
        x = self.pairs[item][0]
        # print(f'x------>{x}')
        # 取出法文
        y = self.pairs[item][1]
        # print(f'y------>{y}')
        # 将样本x进行张量化表示
        # 将输入文本按空格分割成单词列表，并将每个英文单词转换为其对应的索引值
        # 最后在序列末尾添加结束标记（EOS_token），该标记在编码器阶段可选择性添加
        x2index = [english_word2index[word] for word in x.split(' ')]
        x2index.append(EOS_token)  # 可加可不加（编码器阶段）
        # 将上述的结果张量化
        tensor_x = torch.tensor(x2index, dtype=torch.long, device=device)
        # print(f'tensor_x--》{tensor_x}')
        # 将y进行张量化表示
        y2index = [french_word2index[word] for word in y.split(' ')]
        y2index.append(EOS_token)  # 一定加
        # 将上述的结果张量化
        tensor_y = torch.tensor(y2index, dtype=torch.long, device=device)
        return tensor_x, tensor_y


# todo:4.实例化dataloader

def get_dataloader():
    # 实例dataset对象
    seq_dataset = SeqDataset(eng_fra_pairs)
    # 实例化dataloader
    train_dataloader = DataLoader(dataset=seq_dataset,
                                  batch_size=1,
                                  shuffle=True)
    return train_dataloader


# todo:5. 定义GRU编码器
class EncoderGRU(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size):
        super().__init__()
        # eng_vocab_size:英文单词的总个数，需要被embedding单词的数量
        self.eng_vocab_size = eng_vocab_size
        # hidden_size:代表单词的词嵌入维度
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embed = nn.Embedding(eng_vocab_size, hidden_size)

        # 定义GRU层:注意：这里输入和输出维度一致，并且设置了batch_first=True,意味这gru模型的输入是：【batch_size, seq_len, embedding_dim】
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, h0):
        # x--》来自于dataloader，形状为--》[batch_size, seq_len]-->[1, 8]
        # h0-->初始化的gru模型的隐藏层张量的结果--》[1, 1, 256]
        # 再将x送入Gru模型之前，一定要转换为三维的张量，所以x-->[1, 8]-->[1, 8, 256]
        embed_x = self.embed(x)
        # 将embed_x和h0送入gru模型
        #  output-->[1, 8, 256]; hn-->[1, 1, 256]
        output, hn = self.gru(embed_x, h0)
        return output, hn

    def inithidden(self):
        # 注意：需要将张量放到GPU上
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo: 6.定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self, fre_vocab_size, hidden_size):
        super().__init__()
        # fre_vocab_size：代表：法文单词的总个数:4345
        self.fre_vocab_size = fre_vocab_size
        # hidden_size：代表：词嵌入的维度:256
        self.hidden_size = hidden_size
        # 定义Embedding 层
        self.embed = nn.Embedding(fre_vocab_size, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义输出层
        self.out = nn.Linear(hidden_size, fre_vocab_size)

    def forward(self, y0, h0):
        # y0--》来自于dataloader，形状为--》[batch_size, 1]-->[1, 1]
        # h0-->初始化的gru模型的隐藏层张量的结果--》[1, 1, 256]
        # 1.再将y0送入Gru模型之前，一定要转换为三维的张量，所以y0-->[1, 1]-->[1, 1, 256]
        embed_y0 = self.embed(y0)
        # 2.将上述的embed_y0经过relu激活函数，可以防止过拟合
        relu_y0 = F.relu(embed_y0)
        # 3.将relu之后的结果送入gru模型:output-->[1, 1, 256]
        output, hn = self.gru(relu_y0, h0)
        # 4.将gru模型的输出结果送入输出层，但是需要降维
        # result ===>[1, fre_vocab_size]-->[1, 4345]
        result = self.out(output[0])
        # 5.将上述的result进行log_softmax
        return F.log_softmax(result, dim=-1), hn

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def test_decoder():
    # 1.获取训练数据集
    train_dataloader = get_dataloader()
    # 2.实例化encoder
    eng_vocab_size = len(english_word2index)
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # 需要把模型放到GPU上
    encoder = encoder.to(device=device)
    # 3. 实例化解码器对象
    fre_vocab_size = french_word_n
    hidden_size = 256
    decoder = DecoderGRU(fre_vocab_size, hidden_size)
    decoder = decoder.to(device=device)
    # 4.开始将数据送入seq2seq架构得到结果
    for x, y in train_dataloader:
        print(f'x---》{x.shape}')
        print(f'y---》{y.shape}')
        print(f'y---》{y}')
        # 将x送入编码器得到编码器的结果
        h0 = encoder.inithidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        print(f'encoder_output---》{encoder_output.shape}')
        print(f'encoder_hidden---》{encoder_hidden.shape}')
        hidden = encoder_hidden
        # 开始解码：注意，一定是一个词一个词去解码
        for idx in range(y.shape[1]):
            temp_vector = y[0][idx].view(1, -1)
            output, hidden = decoder(temp_vector, hidden)
            print(f'output--》{output.shape}')
        break


# todo: 7.定义带attention的解码器

class AttentionDecoder(nn.Module):
    def __init__(self, french_vocab_size, hidden_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super().__init__()
        # frech_vocab_size：法文单词的总个数
        self.french_vocab_size = french_vocab_size
        # hidden_size：词嵌入的维度
        self.hidden_size = hidden_size
        # dropout_p：随机失活的系数
        self.dropout_p = dropout_p
        # max_len：最大句子长度
        self.max_len = max_len
        # 定义Embedding层：num_embeddings=frech_vocab_size, embedding_dim=hidden_size
        self.embed = nn.Embedding(french_vocab_size, hidden_size)
        # 定义第一个全连接层：计算注意力权重分数
        self.atten = nn.Linear(2 * hidden_size, max_len)
        # 定义第二个全连接层：让注意力的结果按照指定尺寸输出
        self.atten_combin = nn.Linear(2 * hidden_size, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义第三个全连接层：输出层，
        self.out = nn.Linear(hidden_size, french_vocab_size)
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q, K, V):
        # Q代表：当前解码时，预测出的上一个单词（最开始的时候代表：SOS）:[1, 1]
        # K代表：解码器上一层的隐藏层输出结果（最开始的时候是编码器最后一个单词的隐藏层张量的结果）[1, 1, hidden_size]-->[1,1,256]
        # V代表：编码器每个时间步的隐藏层的输出结果--》但是我们规定了最大句子长度--》[max_len, hidden_size]-->[10, 256]
        # 1.需要将Q输入Embedding层: embed_x-->[1, 1, 256]
        embed_x = self.embed(Q)
        # 2. 对embed_x进行随机失活：防止过拟合dropout_x:[1, 1, 256]
        dropout_x = self.dropout(embed_x)
        # 3.按照注意力的计算步骤，实现最终注意力的计算
        # 3.1 选择第一种注意力计算规则，实现Q\K|V的运算
        # 3.1.1 将Q（dropout_x）和K进行拼接--》[1, 1, 512]-->经过Linear层(512, max_len)-->[1, 1, 10]
        atten_weight = F.softmax(self.atten(torch.cat((dropout_x, K), dim=-1)), dim=-1)
        # 3.1.2 将atten_weight[1, 1, 10]和V[10, 256]进行矩阵乘法运算得到结果:temp_vc-->[1,1,256]
        temp_vc = torch.bmm(atten_weight, V.unsqueeze(dim=0))
        # 3.2 将上述第一步计算的结果和Q(dropout_x)进行拼接:[1, 1, 512]
        cat_vc = torch.cat((dropout_x, temp_vc), dim=-1)
        # 3.3 将上述拼接之后的结果按照指定尺寸输出attention_output:[1, 1, 256]
        attention_output = F.relu(self.atten_combin(cat_vc))
        # 4. 将attention_output以及K(hidden)送入GRU模型output-->[1, 1, 256]
        output, hidden = self.gru(attention_output, K)
        # 5.将output降维送入输出层:result---[1, 4345]
        result = self.out(output[0])
        return F.log_softmax(result), hidden, atten_weight


def test_attentionDecoder():
    # 1.获取训练数据集
    train_dataloader = get_dataloader()
    # 2.实例化encoder
    eng_vocab_size = len(english_word2index)
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # 需要把模型放到GPU上
    encoder = encoder.to(device=device)
    # 3. 实例化Attention解码器对象
    fre_vocab_size = french_word_n
    hidden_size = 256
    decoder = AttentionDecoder(fre_vocab_size, hidden_size)
    attention_decoder = decoder.to(device=device)
    # 4.开始将数据送入seq2seq架构得到结果
    for x, y in train_dataloader:
        print(f'x---》{x.shape}')
        print(f'y---》{y.shape}')
        print(f'y---》{y}')
        # 将x送入编码器得到编码器的结果
        h0 = encoder.inithidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        # print(f'encoder_output---》{encoder_output}')
        # print(f'encoder_output---》{encoder_output.shape}')
        # print(f'encoder_hidden---》{encoder_hidden.shape}')
        # # 定义中间语意张量C
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        # 将真实的x编码后的结果赋值给encoder_output_c，其余多余的为0
        # encoder_output-->[1, 5, 256]
        for idx in range(encoder_output.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        # print(f'encoder_output_c--》{encoder_output_c}')
        # 解码：一个token一个token去解码
        hidden = encoder_hidden
        for j in range(y.shape[1]):
            temp_vec = y[0, j].view(1, -1)
            output, hidden, atten_weight = attention_decoder(Q=temp_vec, K=hidden, V=encoder_output_c)
            print(f'output->{output.shape}')
            print(f'hidden->{hidden.shape}')
            print(f'atten_weight->{atten_weight.shape}')


# todo:8.定义模型的训练函数
# 8.1 定义模型训练的超参数
my_lr = 1e-4
epochs = 1
teacher_forcing_ratio = 0.5


# 8.2 定义模型的训练函数

def train_seqseq():
    # 1.获取数据
    train_dataloader = get_dataloader()
    # 2.实例化编码器对象
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder = encoder.to(device=device)
    # 3.实例化带Attention的解码器对象
    french_vocab_size = french_word_n
    hidden_size = 256
    atten_decoder = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # 4.实例化优化器对象
    encoder_adam = optim.Adam(encoder.parameters(), lr=my_lr)
    atten_decoder_adam = optim.Adam(atten_decoder.parameters(), lr=my_lr)
    # 5.实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 6. 定义存储损失的列表
    plot_loss_list = []
    # 7. 开始外部循环
    for epoch_idx in range(1, epochs + 1):
        # 内部定义一些训练日志的参数
        print_loss_total, plot_loss_total = 0.0, 0.0
        start_time = time.time()
        # 7.1 开始内部迭代循环
        for item, (x, y) in enumerate(tqdm(train_dataloader), start=1):
            # 开始调用内部迭代的函数
            # print(f'x---.{x}')
            # print(f'y---.{y}')
            my_loss = train_iter(x, y, encoder, atten_decoder, encoder_adam, atten_decoder_adam, cross_entropy)
            # print(f'my_loss--》{my_loss}')
            print_loss_total += my_loss
            plot_loss_total += my_loss
            # 每隔1000步打印损失日志
            if item % 1000 == 0:
                # 计算平均损失
                avg_loss = print_loss_total / 1000
                # print_loss_total重新初始化为0
                print_loss_total = 0.0
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, avg_loss, time.time() - start_time))

            # 每隔100步，保存平均损失，画图
            if item % 100 == 0:
                plot_avg_loss = plot_loss_total / 100
                plot_loss_list.append(plot_avg_loss)
                plot_loss_total = 0.0

        torch.save(encoder.state_dict(), './save_model/ai23_seq2seq_encode_%d.pth' % epochs)
        torch.save(atten_decoder.state_dict(), './save_model/ai23_seq2seq_decode_%d.pth' % epochs)
    # 8.绘图
    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('ai23_seq2seq_loss.png')
    plt.show()


# 8.3 定义模型内部迭代训练函数

def train_iter(x, y, encoder, atten_decoder, encoder_adam, atten_decoder_adam, cross_entropy):
    # x---》代表英文原始数据的输入--》[batch_size, seq_len]-->[1, 6]
    # y---》代表法文原始数据的输入--》[batch_size, seq_len]-->[1, 8]
    # encoder:编码器
    # atten_decoder： 带Attention的解码器
    # encoder_adam： 编码器优化器
    # atten_decoder_adam： 带Attention的解码器优化器
    # cross_entropy：损失函数对象
    # 1. 需要将x送入编码器得到编码器之后的结果:encoder_output-->[1, 6, 256];encoder_hidden-->[1, 1, 256]
    h0 = encoder.inithidden()
    encoder_output, encoder_hidden = encoder(x, h0)
    # 2.进行解码器参数的准备:
    # 2.1 encoder_output_c-->就是代表的Value
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    # 将真实的编码结果进行赋值
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]
    # print(f'encoder_output_c-->{encoder_output_c.shape}')
    # 2.2: encoder_hidden:代表解码器上一时间步的隐藏层输出，这里其实就是代表：Key
    # 这里解码器的第一个时间步的隐藏层输入用编码器的最后一个时间步的隐藏层输出结果初始化
    decoder_hidden = encoder_hidden
    # print(f'decoder_hidden--》{decoder_hidden.shape}')
    # 2.3 :定义解码器开始解码的第一个字符为SOS，代表Query
    input_y = torch.tensor([[SOS_token]], device=device)
    # print(f'input_y--》{input_y}')

    # 3.定义变量
    my_loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 4.将数据送入解码器
    if use_teacher_forcing:
        # 真实样本法文句子有多长，就要遍历多少次
        for idx in range(y_len):
            # 获取模型预测的结果
            output_y, decoder_hidden, atten_weight = atten_decoder(Q=input_y, K=decoder_hidden, V=encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # 获取真实的标签结果
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)
            # 用真实的label当作下一个输入
            input_y = y[0][idx].view(1, -1)
    else:
        # 真实样本法文句子有多长，就要遍历多少次
        for idx in range(y_len):
            # 获取模型预测的结果
            output_y, decoder_hidden, atten_weight = atten_decoder(Q=input_y, K=decoder_hidden, V=encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # 获取真实的标签结果
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)
            topv, topi = torch.topk(output_y, k=1)
            # print(f'topi--》{topi}')
            # print(f'topi--》{topi.item()}')
            # print(f'topv--》{topv}')
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()

    # 5.梯度清零
    encoder_adam.zero_grad()
    atten_decoder_adam.zero_grad()
    # 6.反向传播
    my_loss.backward()
    # 7.参数更新
    encoder_adam.step()
    atten_decoder_adam.step()
    return my_loss.item() / y_len


# todo: 9.实现评估函数的应用
# 准备好已经训练好的模型的路径
encoder_path = './save_model/ai23_seq2seq_encode_1.pth'
decoder_path = './save_model/ai23_seq2seq_decode_1.pth'


def seq2seq_evaluate(tensor_x, encoder_model, decoder_model):
    with torch.no_grad():
        # 1. 需要将tensor_x送入编码器得到编码器之后的结果:encoder_output-->[1, 6, 256];encoder_hidden-->[1, 1, 256]
        h0 = encoder_model.inithidden()
        encoder_output, encoder_hidden = encoder_model(tensor_x, h0)
        # 2.进行解码器参数的准备:
        # 2.1 encoder_output_c-->就是代表的Value
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder_model.hidden_size, device=device)
        # 将真实的编码结果进行赋值
        for idx in range(tensor_x.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        # print(f'encoder_output_c-->{encoder_output_c.shape}')
        # 2.2: encoder_hidden:代表解码器上一时间步的隐藏层输出，这里其实就是代表：Key
        # 这里解码器的第一个时间步的隐藏层输入用编码器的最后一个时间步的隐藏层输出结果初始化
        decoder_hidden = encoder_hidden
        # print(f'decoder_hidden--》{decoder_hidden.shape}')
        # 2.3 :定义解码器开始解码的第一个字符为SOS，代表Query
        input_y = torch.tensor([[SOS_token]], device=device)
        # print(f'input_y--》{input_y}')

        # 定义存储解码出法文单词的列表
        decoded_list = []
        # 定义存储解码出每个token的注意力权重的张量
        decoded_attention = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 开始解码过程
        # idx = 0
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = decoder_model(input_y, decoder_hidden, encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # print(f'atten_weight--》{atten_weight.shape}')
            # 将每一步预测的注意力的权重进行赋值
            # decoded_attention[idx] = atten_weight[0, 0]
            decoded_attention[idx] = atten_weight
            # 获得最大预测值的概率值以及对应的索引
            topv, topi = torch.topk(output_y, k=1)
            # print(f'topi--》{topi}')
            if topi.item() == EOS_token:
                decoded_list.append('<EOS>')
                break
            else:
                decoded_list.append(french_index2word[topi.item()])
            # print(f'decoded_list--》{decoded_list}')
            input_y = topi
    return decoded_list, decoded_attention[:idx + 1]


def use_evaluate():
    # 实例化编码器对象，并且加载训练好的模型参数
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder_model = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder_model.load_state_dict(torch.load(encoder_path))
    encoder_model.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f'encoder_model--》{encoder_model}')
    # for parameter in encoder_model.parameters():
    #     print(parameter.device)
    #     break

    # 实例化解码器对象，并且加载训练好的模型参数
    french_vocab_size = french_word_n
    hidden_size = 256
    decoder_model = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # decoder_model.load_state_dict(torch.load(decoder_path))
    decoder_model.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    print(f'decoder_model--》{decoder_model}')
    # 准备测试的语料
    my_samplepairs = [['i m impressed with your french .', 'je suis impressionne par votre francais .'],
                      ['i m more than a friend .', 'je suis plus qu une amie .'],
                      ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    print('my_samplepairs--->', len(my_samplepairs))

    # 将遍历每一个测试样本，将原始的英文句子送入模型得到预测结果，并且和真实的法文标签做对比
    for item, pair in enumerate(my_samplepairs):
        # print(f'pair--》{pair}')
        x = pair[0]
        y = pair[1]
        # 将x进行张量化
        temp_x = [english_word2index[word] for word in x.split(' ')]
        temp_x.append(EOS_token)
        tensor_x = torch.tensor(temp_x, dtype=torch.long, device=device).view(1, -1)
        # print(f'tensor_x--》{tensor_x}')
        # 将张量化的x送入评估函数
        deocder_words, attention = seq2seq_evaluate(tensor_x, encoder_model, decoder_model)
        # print(f'deocder_words--》{deocder_words}')
        # print(f'attention--》{attention.shape}')
        predit_y = ' '.join(deocder_words)
        print('最终的预测结果')
        print(f'x--->{x}')
        print(f'y--->{y}')
        print(f'predit_y--->{predit_y}')


def show_attention():
    # 实例化编码器对象，并且加载训练好的模型参数
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder_model = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder_model.load_state_dict(torch.load(encoder_path))
    encoder_model.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f'encoder_model--》{encoder_model}')
    # for parameter in encoder_model.parameters():
    #     print(parameter.device)
    #     break

    # 实例化解码器对象，并且加载训练好的模型参数
    french_vocab_size = french_word_n
    hidden_size = 256
    decoder_model = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # decoder_model.load_state_dict(torch.load(decoder_path))
    decoder_model.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    print(f'decoder_model--》{decoder_model}')
    # 准备测试的语料
    sentence = "we are both teachers ."

    temp_x = [english_word2index[word] for word in sentence.split(' ')]
    temp_x.append(EOS_token)
    tensor_x = torch.tensor(temp_x, dtype=torch.long, device=device).view(1, -1)
    # print(f'tensor_x--》{tensor_x}')
    # 将张量化的x送入评估函数
    deocder_words, attention = seq2seq_evaluate(tensor_x, encoder_model, decoder_model)
    # print(f'deocder_words--》{deocder_words}')
    # print(f'attention--》{attention.shape}')
    predit_y = ' '.join(deocder_words)
    print('最终的预测结果')
    print(f'predit_y--->{predit_y}')
    plt.matshow(attention.detach().numpy())
    plt.savefig('ai23—_attention.png')
    plt.show()


if __name__ == '__main__':
    # eng_vocab_size = len(english_word2index)
    # hidden_size = 256
    # encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # # 需要把模型放到GPU上
    # encoder = encoder.to(device=device)
    # print(encoder)
    # train_dataloader = get_dataloader()
    # # print(len(train_dataloader))
    # for x, y in train_dataloader:
    #     h0 = encoder.inithidden()
    #     output, hn = encoder(x, h0)
    #     print(f'output--》{output.shape}')
    #     print(f'hn--》{hn.shape}')
    #     break
    # 实例化解码器对象
    # fre_vocab_size = french_word_n
    # hidden_size = 256
    # decoder = DecoderGRU(fre_vocab_size, hidden_size)
    # print(decoder)
    # test_decoder()
    # french_vocab_size = french_word_n
    # hidden_size = 256
    # atten_decoder = AttentionDecoder(french_vocab_size, hidden_size)
    # print(atten_decoder)
    # test_attentionDecoder()
    # train_seqseq()
    use_evaluate()
    # show_attention()

""" train_seqseq() 训练时间
1.参数：
    设备名称	HC-YBWJRDPVHOSA
    处理器	Intel(R) Core(TM) i9-14900HX   2.20 GHz
    机带 RAM	32.0 GB (31.6 GB 可用)
    设备 ID	59FA5770-4F92-47A5-B12E-D7232E870CAB
    产品 ID	00330-80000-00000-AA765
    系统类型	64 位操作系统, 基于 x64 的处理器
    笔和触控	为 10 触摸点提供触控支持
2.使用CPU进行训练
3.训练批次：epochs = 1
4.学习率 my_lr = 1e-4
5. teacher_forcing_ratio = 0.5
6.训练时间： 41:31
7.损失1.840459
D:\software\conda_base\envs\nlpenv\python.exe D:\code\study\nlp\05_RNN\dm02_seq2seq.py 
  0%|          | 0/63594 [00:00<?, ?it/s]D:\code\study\nlp\05_RNN\dm02_seq2seq.py:278: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(result), hidden, atten_weight
  2%|▏         | 1000/63594 [00:27<30:05, 34.66it/s]轮次1  损失4.110199 时间:27
  3%|▎         | 2000/63594 [00:56<29:07, 35.24it/s]轮次1  损失3.615103 时间:56
  5%|▍         | 2998/63594 [01:25<30:45, 32.84it/s]轮次1  损失3.488359 时间:86
  6%|▋         | 3998/63594 [01:55<31:11, 31.84it/s]轮次1  损失3.351566 时间:115
  8%|▊         | 4999/63594 [02:25<28:23, 34.40it/s]轮次1  损失3.255308 时间:145
  9%|▉         | 5998/63594 [02:55<28:21, 33.85it/s]轮次1  损失3.169401 时间:175
 11%|█         | 6998/63594 [03:25<28:29, 33.10it/s]轮次1  损失3.153084 时间:205
 13%|█▎        | 8000/63594 [04:03<37:16, 24.86it/s]轮次1  损失3.111183 时间:243
 14%|█▍        | 8998/63594 [04:39<36:36, 24.86it/s]轮次1  损失2.947042 时间:279
 16%|█▌        | 9999/63594 [05:20<38:03, 23.47it/s]轮次1  损失2.952071 时间:320
 17%|█▋        | 10998/63594 [05:59<34:47, 25.20it/s]轮次1  损失2.860942 时间:359
 19%|█▉        | 11998/63594 [06:38<33:43, 25.49it/s]轮次1  损失2.954845 时间:398
 20%|██        | 12998/63594 [07:19<33:33, 25.13it/s]轮次1  损失2.837321 时间:439
 22%|██▏       | 14000/63594 [08:01<34:19, 24.08it/s]轮次1  损失2.856541 时间:481
 24%|██▎       | 14999/63594 [08:44<35:03, 23.10it/s]轮次1  损失2.731876 时间:524
 25%|██▌       | 15999/63594 [09:27<33:56, 23.37it/s]轮次1  损失2.758936 时间:567
 27%|██▋       | 17000/63594 [10:10<32:36, 23.81it/s]轮次1  损失2.692836 时间:610
 28%|██▊       | 17999/63594 [10:53<33:26, 22.72it/s]轮次1  损失2.695294 时间:653
 30%|██▉       | 18998/63594 [11:37<32:49, 22.64it/s]轮次1  损失2.649986 时间:697
 31%|███▏      | 20000/63594 [12:20<31:53, 22.79it/s]轮次1  损失2.632946 时间:740
 33%|███▎      | 20999/63594 [13:03<27:40, 25.64it/s]轮次1  损失2.567700 时间:784
 35%|███▍      | 21998/63594 [13:45<27:43, 25.01it/s]轮次1  损失2.639046 时间:825
 36%|███▌      | 23000/63594 [14:29<28:54, 23.41it/s]轮次1  损失2.545250 时间:869
 38%|███▊      | 23999/63594 [15:12<28:15, 23.36it/s]轮次1  损失2.459038 时间:912
 39%|███▉      | 24998/63594 [15:56<28:49, 22.32it/s]轮次1  损失2.465057 时间:956
 41%|████      | 26000/63594 [16:39<28:01, 22.36it/s]轮次1  损失2.516123 时间:999
 42%|████▏     | 26999/63594 [17:23<27:45, 21.98it/s]轮次1  损失2.437782 时间:1043
 44%|████▍     | 28000/63594 [18:07<25:51, 22.94it/s]轮次1  损失2.393149 时间:1087
 46%|████▌     | 28999/63594 [18:50<25:34, 22.54it/s]轮次1  损失2.346081 时间:1130
 47%|████▋     | 29998/63594 [19:33<23:09, 24.18it/s]轮次1  损失2.307461 时间:1173
 49%|████▊     | 30998/63594 [20:14<16:14, 33.44it/s]轮次1  损失2.354694 时间:1214
 50%|█████     | 32000/63594 [20:54<22:37, 23.28it/s]轮次1  损失2.248268 时间:1254
 52%|█████▏    | 32999/63594 [21:38<22:26, 22.73it/s]轮次1  损失2.292815 时间:1298
 53%|█████▎    | 33998/63594 [22:20<21:08, 23.33it/s]轮次1  损失2.216743 时间:1340
 55%|█████▌    | 35000/63594 [23:03<21:58, 21.69it/s]轮次1  损失2.270199 时间:1383
 57%|█████▋    | 35999/63594 [23:46<19:05, 24.08it/s]轮次1  损失2.195169 时间:1426
 58%|█████▊    | 36998/63594 [24:28<18:33, 23.88it/s]轮次1  损失2.172274 时间:1468
 60%|█████▉    | 38000/63594 [25:11<18:09, 23.50it/s]轮次1  损失2.189546 时间:1511
 61%|██████▏   | 38999/63594 [25:54<18:37, 22.01it/s]轮次1  损失2.181996 时间:1554
 63%|██████▎   | 39998/63594 [26:38<14:54, 26.39it/s]轮次1  损失2.151927 时间:1598
 64%|██████▍   | 40997/63594 [27:09<11:47, 31.95it/s]轮次1  损失2.135663 时间:1629
 66%|██████▌   | 41999/63594 [27:41<11:06, 32.38it/s]轮次1  损失2.105531 时间:1661
 68%|██████▊   | 42997/63594 [28:12<11:30, 29.81it/s]轮次1  损失2.112397 时间:1692
 69%|██████▉   | 44000/63594 [28:44<11:05, 29.43it/s]轮次1  损失2.098468 时间:1724
 71%|███████   | 44999/63594 [29:15<09:56, 31.19it/s]轮次1  损失2.106733 时间:1755
 72%|███████▏  | 46000/63594 [29:46<08:41, 33.75it/s]轮次1  损失2.006586 时间:1786
 74%|███████▍  | 47000/63594 [30:18<08:50, 31.28it/s]轮次1  损失2.015471 时间:1818
 75%|███████▌  | 47999/63594 [30:50<07:52, 32.99it/s]轮次1  损失2.071022 时间:1850
 77%|███████▋  | 48999/63594 [31:26<10:22, 23.45it/s]轮次1  损失2.003260 时间:1886
 79%|███████▊  | 49998/63594 [32:02<07:06, 31.86it/s]轮次1  损失2.037119 时间:1922
 80%|████████  | 51000/63594 [32:42<08:47, 23.88it/s]轮次1  损失1.940661 时间:1962
 82%|████████▏ | 51998/63594 [33:24<07:41, 25.11it/s]轮次1  损失1.970081 时间:2004
 83%|████████▎ | 52999/63594 [34:01<07:32, 23.40it/s]轮次1  损失1.954238 时间:2041
 85%|████████▍ | 53998/63594 [34:44<07:03, 22.64it/s]轮次1  损失1.938302 时间:2084
 86%|████████▋ | 55000/63594 [35:27<06:22, 22.46it/s]轮次1  损失1.855416 时间:2127
 88%|████████▊ | 55999/63594 [36:10<05:10, 24.48it/s]轮次1  损失1.915089 时间:2170
 90%|████████▉ | 56998/63594 [36:53<04:41, 23.41it/s]轮次1  损失1.939793 时间:2213
 91%|█████████ | 58000/63594 [37:36<04:05, 22.83it/s]轮次1  损失1.872065 时间:2256
 93%|█████████▎| 58999/63594 [38:19<03:22, 22.65it/s]轮次1  损失1.867871 时间:2299
 94%|█████████▍| 59998/63594 [39:02<02:40, 22.34it/s]轮次1  损失1.788892 时间:2342
 96%|█████████▌| 61000/63594 [39:46<01:53, 22.78it/s]轮次1  损失1.834171 时间:2386
 97%|█████████▋| 61999/63594 [40:24<00:55, 28.69it/s]轮次1  损失1.819260 时间:2424
 99%|█████████▉| 62999/63594 [41:06<00:25, 23.65it/s]轮次1  损失1.840459 时间:2466
100%|██████████| 63594/63594 [41:31<00:00, 25.52it/s]

进程已结束，退出代码为 0

"""
