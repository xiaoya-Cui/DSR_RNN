import torch.nn as nn
import torch.nn.functional as F
import torch

class DSRRNN(nn.Module):
    def __init__(self,
                 operators,
                 hidden_size,
                 device,
                 min_length,
                 max_length,
                 type,
                 num_layers,
                 dropout):
        super(DSRRNN,self).__init__()

        self.input_size = 2*len(operators)
        self.hidden_size = hidden_size
        self.output_size = len(operators)
        self.num_layers = num_layers
        self.dropout = dropout
        self.operators = operators
        self.device = device
        self.type = type

        self.init_input = nn.Parameter(data=torch.rand(1,self.input_size),requires_grad=True).to(self.device)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers,self.hidden_size), requires_grad=True).to(self.device)

        self.min_length = min_length
        self.max_length = max_length

        if (self.type == 'rnn'):
            self.rnn = nn.RNN(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size,self.output_size).to(self.device)
        elif (self.type == 'lstm'):
            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                proj_size=self.output_size,
                dropout=self.dropout
            ).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size), requires_grad=True).to(self.device)
        elif (self.type == 'gru'):
            self.gru = nn.GRU(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    def sample_sequence(self,n,min_length,max_length):
        sequences = torch.zeros((n,0))
        entropies = torch.zeros((n,0))
        log_probs = torch.zeros((n,0))

        sequences_mask = torch.ones((n,1))

        input_tensor = self.init_input.repeat(n,1)
        hidden_tensor = self.init_hidden.repeat(n,1)
        if(self.type == 'lstm'):
            hidden_lstm = self.init_hidden_lstm.repeat(n,1)

        counters = torch.ones(n)
        lengths = torch.zeros(n)

        #生成表达式序列
        while(sequences_mask.all(dim=1).any()):
            if(self.type == 'rnn'):
                output,hidden_tensor = self.forward(input_tensor,hidden_tensor)
            elif(self.type == 'lstm'):
                output,hidden_tensor,hidden_lstm = self.forward(input_tensor,hidden_tensor,hidden_lstm)
            elif(self.type == 'gru'):
                output,hidden_tensor = self.forward(input_tensor,hidden_tensor)
            #添加限制/归一化概率
            output = self.apply_constraints(output,counters,lengths,sequences)
            output = output/torch.sum(output,axis=1)[:,None]

            #采样 将token添加到sequences
            dist = torch.distributions.Categorical(output)
            token = dist.sample()
            sequences = torch.cat((sequences,token.unsqueeze(1)),axis=1)
            lengths += 1

            #将当前token的对数概率添加到log_probs中，熵添加到entropies
            log_probs = torch.cat((log_probs,dist.log_prob(token).unsqueeze(1)),axis=1)
            entropies = torch.cat((entropies,dist.entropy().unsqueeze(1)),axis=1)

            #根据当前操作符类型更新计数器
            counters -= 1
            counters += torch.isin(token,self.operators.arity_two).long() * 2
            counters += torch.isin(token,self.operators.arity_one).long() * 1

            #对下一个采样的token，确定哪些位置可以生成新元素,哪些位置已经被填充
            #下一个位置counters>0并且sequences_mask某行所有位置都为True
            sequences_mask = torch.cat(
                (sequences_mask,torch.bitwise_and((counters>0)[:,None],sequences_mask.all(dim=1)[:,None])),
                axis=1)

            #计算下一个token父节点兄弟节点，组成输入
            parent_sibling = self.get_parent_sibling(sequences,lengths)
            input_tensor = self.get_next_input(parent_sibling)

        entropies = torch.sum(entropies * (sequences_mask[:,:-1]).long(),axis=1)
        log_probs = torch.sum(log_probs * (sequences_mask[:,:-1]).long(),axis=1)
        sequences_lengths = torch.sum(sequences_mask.long(),axis=1)
        #有效长度

        return sequences,sequences_lengths,entropies,log_probs

    def forward(self,input,hidden,hidden_lstm=None):
        """
        输入是父节点，子节点 input:[2000,24] hidden:[2000,250]
         [batch_size,input_size]  [batch_size*num_layers,hidden_size]
        """
        if(self.type == 'rnn'):
            output,hidden = self.rnn(input.unsqueeze(1).float(),hidden.unsqueeze(0))
            #input:[batch_size,1,input_size] hidden:[1,batch_size*num_layers,hidden_size]
            #output:[batch_size,1,hidden_size] hidden:[1,batch_size,hidden_size]
            output = output[:,0,:]
            #[batch_size,hieedn_size]
            output = self.projection_layer(output)
            #[batch_size,12(output_size)]
            output = self.activation(output)
            return  output,hidden[0,:]
            #hidden[batch_size,hidden_size]
        elif (self.type == 'lstm'):
            output,(hn,cn) = self.lstm(input[:,None].float(),(hidden_lstm[None,:],hidden[None,:]))
            output = output[:,0,:]
            output = self.activation(output)
            return output,cn[0,:],hn[0,:]
        elif (self.type == 'gru'):
            output,hn = self.gru(input[:,None].float(),hidden[None,:])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]

    def apply_constraints(self, output, counters, lengths, sequences):
        """
        output:[batch_size,output_size]
        counters:batch_size*[1]
        lengths:batch_size*[0]
        sequences:[batch_size,0]
        """
        #添加极小数
        epsilon = torch.ones(output.shape) * 1e-20
        output = output + epsilon.to(self.device)

        #（1）最大长度限制 [min_length,max_length]
        min_boolean_mask = (counters + lengths >= torch.ones(counters.shape) * self.min_length).long()[:, None]
        # 满足最小长度的置为1，其他不变[2000,1]
        min_length_mask = torch.max(self.operators.nonzero_arity_mask[None, :], min_boolean_mask)
        # 只有非零操作符需要满足此限制broadcost [2000,12]tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],...]
        output = torch.minimum(output, min_length_mask)
        # 满足最小长度要求的概率被保留，不满足为0

        #（2）最小长度限制
        max_boolean_mask = (counters + lengths <= torch.ones(counters.shape) * (self.max_length - 2)).long()[:,None]
        max_length_mask = torch.max(self.operators.zero_arity_mask[None, :], max_boolean_mask)
        output = torch.minimum(output, max_length_mask)

        #（3）变量限制 last token必须是常数或变量
        nonvar_zeroarity_mask = (
            ~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)).long()
        # 筛选出不满足零元且不是变量的位置 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
        if (lengths[0].item() == 0.0):  # item将tensor转化成标量
            output = torch.minimum(output, nonvar_zeroarity_mask)
        else:
            nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # 对每个维度复制[batch_size,12]
            counter_mask = (counters == 1)
            contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            # [batch_size]  True表示相应序列中没有变量，False有变量
            last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            # 筛选出哪些序列最后一个token是非变量0元操作符
            nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(
                nonvar_zeroarity_mask.shape)).long()
            # 确保在计数器为1且还未生成变量的情况下，只能生成零元操作符。
            output = torch.minimum(output, nonvar_zeroarity_mask)


        # (4) 三角函数限制 内部节点不能是三角函数
        Batch_size, num_operators = output.shape
        funs = self.check_trig(sequences)
        if sequences.size()[1] >= 1:

            trig_tensor = torch.tensor([5, 6, 7])
            last_tokens = sequences[:, -1]

            is_trig_fun = torch.isin(last_tokens, trig_tensor)

            for i in range(Batch_size):
                if is_trig_fun[i]:
                    output[i, [5, 6, 7]] = 0.0
                if funs[i]:
                    output[i, [5, 6, 7]] = 0.0


        #(5)常数限制 一元操作符子节点不能是常数，二元操作符两个子节点不能全是常数
        UNARY_TOKENS = self.operators.arity_one
        BINARY_TOKENS = self.operators.arity_two
        CONST_TOKENS = self.operators.constant_operators

        if sequences.size()[1] >= 1 :
            last_token = sequences[:,-1]
            is_in_unary = torch.isin(last_token,UNARY_TOKENS)
            for j in range(Batch_size):
                if is_in_unary[j]:
                    output[i,10] = 0.0
        if sequences.size()[1] >= 2 :
            last_token = sequences[:,-1]
            last_two_token = sequences[:,-2]
            is_in_binary = torch.isin(last_two_token,BINARY_TOKENS)
            for k in range(Batch_size):
                if last_token[k].item() == 10 and is_in_binary[k]:
                    output[k,10] = 0.0

        return output

    def get_parent_sibling(self,sequences,lengths):
        parent_sibling = torch.ones(sequences.shape[0],2) * -1
        recent = int(lengths[0].item()) - 1
        c = torch.zeros(sequences.shape[0])

        for i in range(recent,-1,-1):
            torch_i = sequences[:,i]

            arity = torch.zeros((lengths.shape[0]))
            arity += torch.isin(torch_i,self.operators.arity_two).long() * 2
            arity += torch.isin(torch_i,self.operators.arity_one).long() * 1

            c += arity
            c -= 1

            c_mask = torch.logical_and(c==0,(parent_sibling==-1).all(axis=1)).unsqueeze(1)
            #筛选出c=0(一元操作符，认为是根节点)并且parents/siblings=-1（还未填充）的token

            i_ip1 = sequences[:,i:i+2]
            i_ip1 = F.pad(i_ip1,(0,1),value=-1)[:,0:2]
            #选择i下当前token和下一个token，先填充-1再切片

            i_ip1 = i_ip1 * c_mask.long()
            #在c_mask为false的位置把i_ip1置0

            parent_sibling = parent_sibling * (~c_mask).long()
            parent_sibling = parent_sibling + i_ip1
            #[batch_size,2]

        recent_nonzero_mask = (~torch.isin(sequences[:,recent],self.operators.arity_zero))[:,None]
        #recent token 有非0元操作符为True,0元操作符False
        parent_sibling = parent_sibling * (~recent_nonzero_mask).long()
        #如果最后一个token非0操作符，清空parents/siblings
        recent_parent_sibling = torch.cat((sequences[:,recent,None],-1*torch.ones((lengths.shape[0],1))),axis=1)
        #nx2 the 2 dimension is :[recent token,-1]
        recent_parent_sibling = recent_parent_sibling * recent_nonzero_mask.long()

        parent_sibling = parent_sibling + recent_parent_sibling

        return  parent_sibling

    def get_next_input(self,parent_sibling):
        parent = torch.abs(parent_sibling[:,0]).long()
        sibling = torch.abs(parent_sibling[:,1]).long()

        parent_onehot = F.one_hot(parent,num_classes=len(self.operators))
        sibling_onehot =F.one_hot(sibling,num_classes=len(self.operators))

        parent_mask = (~(parent_sibling[:,0] == -1)).long()[:,None]
        parent_onehot = parent_onehot * parent_mask
        sibling_mask = (~(parent_sibling[:,1] == -1)).long()[:,None]
        sibling_onehot = sibling_onehot * sibling_mask

        input_tensor = torch.cat((parent_onehot,sibling_onehot),axis=1)
        #[batch_size,2*output_size]

        return  input_tensor

    def check_ind(self, token_ind):
        TRIG_TOKENS = torch.tensor([5, 6, 7])
        BINARY_TOKENS = self.operators.arity_two
        UNARY_TOKENS = self.operators.arity_one
        DEBUG = False
        trig_descendant = False  # 跟踪当前节点是否是三角函数后代
        trig_dangling = None
        # 跟踪三角函数子树中未选择的节点数。
        # 在找到三角函数时初始化为1，然后根据遍历的节点类型进行增减，直到子树中的节点全部被选择。

        for i, name in enumerate(token_ind):
            # 当前是三角函数
            if name in TRIG_TOKENS:
                if trig_descendant:
                    if DEBUG:
                        print('Constrained trig:', token_ind)
                    return True
                trig_dangling = 1
                trig_descendant = True
            # 当前不是三角函数>但是三角函数后代
            elif trig_descendant:  # 当前是三角操作符
                if name in BINARY_TOKENS:
                    trig_dangling += 1
                elif name not in UNARY_TOKENS:  # 当前是变量
                    trig_dangling -= 1
                if trig_dangling == 0:
                    trig_descendant = False
        return False

    def check_trig(self, sequences):
        Batch_size, length = sequences.shape
        funs = []
        for i in range(Batch_size):
            token_ind = sequences[i, :]
            fun = self.check_ind(token_ind)
            funs.append(fun)
        return funs
