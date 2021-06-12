'''
If the model you are instantiating is large, it might need to be split across multiple GPUs. This is easy to do using pytorch. 
'''

from torch import nn

class Network(nn.Module):
    def __init__(self, split_gpus=False):
        super().__init__()
        self.module1 = nn.Linear(100,100,bias=True)
        self.module2 = nn.Linear(100,100,bias=True)
        self.split_gpus = split_gpus
        if split_gpus:            #considering only two gpus
            self.module1.cuda(0)
            self.module2.cuda(1)

    def forward(self, x):
        if self.split_gpus:
            x = x.cuda(0)
        x = self.module1(x)
        if self.split_gpus:
            x = x.cuda(1)
        x = self.module2(x)
        return x

print(Network())