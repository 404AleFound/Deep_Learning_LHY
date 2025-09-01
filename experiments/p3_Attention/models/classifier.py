# classifier.py -- self-attention model

import torch
import torch.nn as nn
import doctest
import math

class Classifier(nn.Module):
    '''
    ---
    TEST
    ---
    >>> x = torch.rand(3, 400, 40)
    >>> model = Classifier(input_dim=40, d_model=80)
    >>> pred = model(x)
    >>> pred.shape
    torch.Size([3, 600])
    '''
    def __init__(self, input_dim=40, d_model=160, n_spks=600, dropout=0.1):
        super(Classifier, self).__init__()
        self.pre_net = nn.Linear(input_dim, d_model)
        # expand the feature dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # self_attn [Q, K, V] shape=(d_model*3, d_model)
            nhead=4, 
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model, n_spks)
        )

    def forward(self, x):
        x = self.pre_net(x)
        new_feature = self.encoder_layer(x)
        new_feature = new_feature.mean(dim=1) # (batch, seq, d_model) -> (batch, d_model)
        out = self.pred_layer(new_feature)
        return out
    
class Multi_Head_Self_Attention(nn.Module):
    '''
    ---
    TEST
    ---
    >>> attention_mask = (torch.tensor([[0, 1],[0, 1],[1, 0]]).unsqueeze(1).unsqueeze(2).expand(3,8,2,2))
    >>> x = torch.rand(3, 2, 128)
    >>> x.shape
    torch.Size([3, 2, 128])
    >>> net = Multi_Head_Self_Attention(128, 8)
    >>> net(x,attention_mask).shape
    torch.Size([3, 2, 128])
    >>> print(net(x,attention_mask).shape == x.shape)
    True
    '''
    def __init__(self, dim_feature, num_heads, attention_dropout=0.1):
        super(Multi_Head_Self_Attention,self).__init__()
        self.dim_feature = dim_feature
        self.num_heads = num_heads
        self.dim_heads = dim_feature // num_heads
        # actually nn.Linear(dim_feature, num_heads * dim_heads)
        self.q_proj = nn.Linear(dim_feature, dim_feature)
        self.k_proj = nn.Linear(dim_feature, dim_feature)
        self.v_proj = nn.Linear(dim_feature, dim_feature)
        self.out_proj = nn.Linear(self.dim_feature, self.dim_feature)

        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x, attention_mask=None):
        # x(b, seq, dim_feature)
        batch, seq_len, _ = x.size()
        # x = x.view(-1, self.dim_feature)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Q, K, V (b, seq, dim_feature) -> (b, seq, num_heads, dim_head)
        q_state = Q.view(batch, seq_len, self.num_heads, self.dim_heads)
        k_state = K.view(batch, seq_len, self.num_heads, self.dim_heads)
        v_state = V.view(batch, seq_len, self.num_heads, self.dim_heads)
        # (b, seq, num_heads, dim_head) -> (b, num_heads, seq, dim_head)
        q_state = q_state.transpose(1,2)
        k_state = k_state.transpose(1,2)
        v_state = v_state.transpose(1,2)

        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1, -2)
        )/math.sqrt(self.dim_heads)
        # attention_weight (b, num_heads, seq, seq)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, 
                float('-inf')
            )
        # print(attention_weight.shape)

        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)
        output_mid = torch.matmul(
            attention_weight, v_state
        ) # (b, num_heads, seq, dim_head) -> (b, seq, num_heads * dim_head)

        output_mid = output_mid.transpose(1,2).contiguous()
        output_mid = output_mid.view(batch, seq_len, -1)
        output = self.out_proj(output_mid)
        return output

if __name__ == '__main__':
    # doct = doctest.testmod(verbose=True)
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = Classifier(input_dim=40, d_model=80).to(device)
    summary(t, (400, 40))