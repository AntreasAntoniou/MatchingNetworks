import torch
import torch.nn.functional as F
import numpy as np
input1 = []

for i in range(1, 6):
    input1.append(np.arange(i, 10+i) * i)

input1 = torch.Tensor(input1).float()
print(input1.shape)
input2 = torch.ones(5, 10).float() * 5
#input2[4, 1] = 0.000001
#
# print(output)
# print(output.shape)

# input1 = input1.view(10, 1)
# input1 = input1.repeat([1, 5])
# input1 = input1.view(10, 5, 1)
# input1 = input1.repeat([1, 1, 5])
# input1 = input1.view(10, 5, 5)
input1 = input1
input2 = input2
output = F.cosine_similarity(input1, input2, dim=1)
print(input1, input2, output)