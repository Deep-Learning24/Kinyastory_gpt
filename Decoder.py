
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
    def forward(self,x):
        return self.embedding(x)

class ShiftedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(ShiftedEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        x = self.embedding(x)
        return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    

class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]

class FeedForward(torch.nn.Module):
    ''' Projection Layer (Fully Connected Layers) '''

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        #2048 x 12288

        self.linear_1   = torch.nn.Linear(d_model, d_ff)
        self.dropout    = torch.nn.Dropout(dropout)
        self.linear_2   = torch.nn.Linear(d_ff, d_model*4)
        self.linear_3   = torch.nn.Linear(d_model*4, d_model)

    def forward(self, x):

        # Apply the first linear layer, GeLU activation, and then dropout
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.dropout(F.relu(self.linear_2(x)))
         # Apply the second linear layer to project the dimension back to d_model
        x = self.linear_3(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.W_O = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def attention(self,Q,K,V,mask=None,dropout=None):
        #Q,K,V: (batch_size,seq_len,d_model)
        #mask: (batch_size,seq_len,seq_len)
        #Q,K,V: (batch_size,num_heads,seq_len,d_k)
        d_k = Q.size(-1)
        attention_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e9)
        attention_scores = self.softmax(attention_scores)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        #attention_scores: (batch_size,num_heads,seq_len,seq_len)
        #V: (batch_size,num_heads,seq_len,d_k)
        return (torch.matmul(attention_scores,V),attention_scores)
    
    def forward(self,Q,K,V,mask=None):
        #Q,K,V: (batch_size,seq_len,d_model)
        #mask: (batch_size,seq_len,seq_len)
        #Q,K,V: (batch_size,num_heads,seq_len,d_k)
        Q = self.W_Q(Q).view(-1,Q.size(1),self.num_heads,self.d_k).transpose(1,2) # (batch_size,seq_len,d_model) -> (batch_size,seq_len,num_heads,d_k)
        K = self.W_K(K).view(-1,K.size(1),self.num_heads,self.d_k).transpose(1,2)
        V = self.W_V(V).view(-1,V.size(1),self.num_heads,self.d_k).transpose(1,2)
        #Q,K: (batch_size,num_heads,seq_len,d_k)
        #scores: (batch_size,num_heads,seq_len,seq_len)
        scores = self.attention(Q,K,V,mask,self.dropout)[0]
        #context: (batch_size,num_heads,seq_len,d_k)
        context = torch.matmul(scores,V)
        #context: (batch_size,seq_len,num_heads,d_k)
        context = context.transpose(1,2).contiguous().view(-1,context.size(2),self.d_model)
        #context: (batch_size,seq_len,d_model)
        context = self.W_O(context)
        return context

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Projection(nn.Module):
    def __init_(self,d_model,vocab_size):
        super(Projection,self).__init__()
        self.linear = nn.Linear(d_model,vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        return self.softmax(self.linear(x))
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model,dropout=dropout)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        #x: (batch_size,seq_len,d_model)
        #mask: (batch_size,seq_len,seq_len)
        x = self.layer_norm_1(x + self.dropout(self.multi_head_attention(x, x, x, mask)))
    
        x = self.layer_norm_2(x + self.dropout(self.feed_forward(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers= layers
        self.norm= LayerNorm()

    def forward(self, inp, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            inp= layer(inp, enc_output, src_mask, tgt_mask)
        return self.norm(inp)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.projection = Projection(d_model, vocab_size)
        self.decoder = Decoder(self.decoder_layers)
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        self.decoder(x, mask)
        x = self.projection(x)
        return x
        
        