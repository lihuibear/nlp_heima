import torch

# a = torch.randn(2, 3, 4)
# 固定的 torch 不随机
# 设置随机种子，固定所有随机操作的结果
# 你可以把42换成任意整数，不同的种子对应不同的固定随机结果
seed = 42
torch.manual_seed(seed)  # 设置CPU的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子

# 此时生成的张量结果会完全固定
a = torch.randn(2, 3, 4)
# print(a)

print(f'a的值->{a}')
print(f'a.shape-{a.shape}')

print("*" * 80)
print("张量的切片")

print(f'a[:]--->{a[:].shape}')  # [2, 3, 4]

print(f'a[1:]--->{a[1:].shape}')  # [1,3,4]

print(f'a[1:,:,]--->{a[1:, :, ].shape}')  # [1,3,4]

# print(f'a[1:,:2,]--->{a[1:,:2,]}')  # [1,2,4]
print(f'a[1:,:2,]--->{a[1:, :2, ].shape}')  # [1,2,4]

print(f'a[1:,2,]--->{a[1:, 2, ]}')  # [1,4]

# print(f'a[0,:1,5]--->{a[0, :1, 5]}')  # 报错
print(f'a[0,:1,3]--->{a[0, :1, 3].shape}')  # [1]

# print(f'a[0,:,:300]--->{a[0, :, :300]}')
print(f'a[0,:,:300]--->{a[0, :, :300].shape}')  # [3,4]
print(f'a[:,2:2] --->{a[:, 2, :2].shape}')  # [2,2]

print(f'a[:1,2] --->{a[:1, 2].shape}')  # [1,4]
print(f'a[0,2:] --->{a[0, 2:].shape}')  # [1,4]
print(f'a[0][2:] --->{a[0][2:].shape}')  # [1,4]

print(f'a[0:,1:,3:] --->{a[0:, 1:, 3:].shape}') # [2,2,1]
