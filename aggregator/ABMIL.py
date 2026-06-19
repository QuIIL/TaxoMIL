import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGated(nn.Module):
    def __init__(self, input_dim=1024, act='relu', bias=False, dropout=0.25, gated=True, K=1, out_dim=1024):
        super().__init__()
        self.L = input_dim
        self.D = 256
        self.K = K
        self.out = out_dim
        self.gated = gated
        self.dropout = dropout

        self.feature = nn.Sequential(
            nn.Linear(input_dim, self.out),
            nn.GELU() if act == 'gelu' else (nn.Tanh() if act == 'tanh' else nn.ReLU()),
            nn.Dropout(self.dropout)
        )
        if gated:
            attn_a = [nn.Linear(self.out, self.D, bias=bias)]
            attn_a += [nn.GELU() if act == 'gelu' else (nn.Tanh() if act == 'tanh' else nn.ReLU())]
            attn_b = [nn.Linear(self.out, self.D, bias=bias), nn.Sigmoid()]
            
            if dropout:
                attn_a += [nn.Dropout(self.dropout)]
                attn_b += [nn.Dropout(self.dropout)]

            self.attention_a = nn.Sequential(*attn_a)
            self.attention_b = nn.Sequential(*attn_b)
            self.attention_c = nn.Linear(self.D, self.K, bias=bias)  # (B,N,D)->(B,N,K)
        else:
            attn = [nn.Linear(self.out, self.D, bias=bias)]
            attn += [nn.GELU() if act == 'gelu' else (nn.Tanh() if act == 'tanh' else nn.ReLU())]
            attn += [nn.Linear(self.D, self.K, bias=bias)]          # (B,N,D)->(B,N,K)

            if dropout:
                attn += [nn.Dropout(self.dropout)]
            self.attention = nn.Sequential(*attn)

    def forward(self, x):
        # x: (B, N, input_dim)
        B, N, _ = x.size()
        x_feat = self.feature(x.view(-1, x.size(-1))).view(B, N, -1)  # (B,N,out)

        if self.gated:
            a = self.attention_a(x_feat)            # (B,N,D)
            b = self.attention_b(x_feat)            # (B,N,D)
            A = self.attention_c(a * b)             # (B,N,K)
        else:
            A = self.attention(x_feat)              # (B,N,K)

        A = A.permute(0, 2, 1).contiguous()         # (B,K,N)
        A = F.softmax(A, dim=-1)                    # (B,K,N)

        x_out = torch.bmm(A, x_feat)                # (B,K,out) = (B,n,1024)
        return x_out
