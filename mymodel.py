import torch.nn as nn
import torch

class MyNet(nn.Module):
    def __init__(
        self,
        ntoken: int,
        ninp: int,
        nhid: int,
        nlayers: int,
        dropout=0.5,
        tie_weights=False,
    ):
        """循环神经网络初始化

        Args:
            ntoken (int): 词表大小(字典大小)
            ninp (int): 模型输入维度,也是embedding层的嵌入大小
            nhid (int): 模型隐藏层h的维度
            nlayers (int): LSTM的堆叠层数。如果有多层LSTM堆叠,前一层最后一个时间步的隐层状态作为后一层的输入
            dropout (float, optional): 随机丢弃参数. Defaults to 0.5.
            tie_weights (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        super(MyNet, self).__init__()
        self.ntoken = ntoken
        self.nhid = nhid
        self.nlayers = nlayers
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp,nhid,nlayers,dropout=dropout)
        self.decoder = nn.Linear(nhid, 2)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """初始化网络参数"""
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self,input):
        # 输入数据进行permute了，所以batchsize是第二个
        batch_size = input.size(1)
        emb = self.drop(self.encoder(input))
        # 初始化h,c的输入
        h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(input.device)
        c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(input.device)
        output,(hn,cn) = self.rnn(emb,(h0,c0))
        # 输出output的shape是[seq_len, batch, hidden_size]，要进行permute交换维度才能输入到线性层
        output = output.permute(1,0,2)
        # 变换之后是[batch,seq_len,hidden_size]
        # 抽取最后一个词的输出作为最终输出
        output = self.decoder(output[:,-1])
        output = self.activation(output)
        return output