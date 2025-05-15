import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert (d_model % num_heads == 0), 'the model dimension must be divisible by the number of heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask != None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('inf'))    
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ V

        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) 

    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        head_outputs = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        concatenated_heads = self.combine_heads(head_outputs)
        output = self.W_o(concatenated_heads)
        output = self.dropout(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # GELU tends to perform better but is more computationally expensive than ReLU
        self.gelu = nn.GELU()
        # self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(d_model, d_ff, bias=True) 
        self.W2 = nn.Linear(d_ff, d_model, bias=True) 

    def forward(self, x):
        x_res = x
        x = self.layer_norm(x)
        x = self.W1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.W2(x)
        x = self.dropout(x)
        
        return x + x_res

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention_layer =  MultiHeadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        x_res = x
        x = self.layer_norm(x)
        x = self.multi_head_attention_layer(x, x, x, mask)

        return x + x_res
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention_layer =  MultiHeadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, Q, K, V, mask=None):
        x_res = V
        Q = self.layer_norm(Q)
        K = self.layer_norm(K)
        V = self.layer_norm(V)
        x = self.multi_head_attention_layer(Q, K, V, mask=mask)

        return x + x_res

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len)
    super().__init__()
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    self.pos = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
    self.div_coeff = torch.exp(torch.arange(0, max_seq_len, 2, dtype=torch.float) * -math.log(10000.0) / d_model)
    
    self.pe = torch.zeros(max_seq_len, d_model)
    self.pe[:, 0::2] = torch.sin(pos * div_coeff)
    self.pe[:, 1::2] = torch.cos(pos * div_coeff)

    self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

######################################## Model ########################################

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.self_attention_block = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.cross_attention_block = CrossAttentionBlock(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff=d_ff * d_model, dropout=dropout)

    def forward(self, x, encoder_output, mask):
        x = self.self_attention_block(x, mask=mask)
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(x)
        x = self.cross_attention_block(Q, K, V)
        x = self.feed_forward(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention_block = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff=d_ff * d_model, dropout=dropout)

    def forward(self, x):
        x = self.self_attention_block(x)
        x = self.feed_forward(x)

        return x

class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_seq_len, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(d_model, max_seq_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout)
        self.W_out = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        def generate_mask():

        def forward(self, x):


       
