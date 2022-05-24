"""
Transformer

modified from the tutorial made by harvard NLP group:
http://nlp.seas.harvard.edu/annotated-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np

from torch.autograd import Variable


#########  Input   ##########
class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        '''
        :param d_model: dimension of embedding
        :param vocab: size of vocabulary
        '''

        super(Embeddings, self).__init__()
        "Simply using nn.Embedding to get the embedding"
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        '''
        :param x: one-hot vector which the input is mapped by the vocabulary
        :return: using the weight sqrt(d_model)
        '''
        embeds = self.lut(x)
        return embeds * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''
        :param d_model: dimension of embedding
        :param dropout: probability to dropout
        :param max_len: max length of sentence
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # positon = [[0], [1], [2],...,[max_len-1]]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # div_term = 1/(10000^([0,2,4,...,d_model]/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        "return 'input embedding + position embedding', add the position information to input embedding"
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


###############   layers for encoder and decoder   #####################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    result = softmax(Q x K / sqrt(d_k)) * V
    """
    d_k = query.size(-1)
    # scores = (Q * K)/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # if using the mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # use softmax
    p_attention = F.softmax(scores, dim = -1)

    # if using the dropout
    if dropout is not None:
        p_attention = dropout(p_attention)

    return torch.matmul(p_attention, value), p_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        MultiHead(Q,K,V) = Concat(head_1,..., head_h)W
        :param h: number of head
        :param d_model: dimension of input word
        :param dropout: probability to dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        # create 4 linear layers, for Q, K, V, and the concat matrix
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attention: the attention tensor we will get, and now is None
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # use unsqueeze to extend the dimension
            mask = mask.unsqueeze(1)
        # nbatches: the first number of query's size, which means the number of samples
        nbatches = query.size(0)

        # make query, key, value to four-dimensional matrix, and the dimension will be 'batch size', 'num of head',
        # 'number of word in a sentence', and 'the dimension of each word's embedding'
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # Compute  attention on all the projected matrix
        x, self.attention = attention(query, key, value, mask=mask,
                                      dropout=self.dropout)
        # use view to get the concat matrix and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        FFN(x)=max(0, xW_1 + b_1)W_2 + b_2
        :param d_model:dimension of the input of the first layer and output of the second layer
        :param d_ff:dimension of the output of the first layer and input of the second layer
        :param dropout:dropout
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: the input from the last layer
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        """
        :param feature_size:dimension of embedding
        """
        super(LayerNorm, self).__init__()
        # create two vectors depend the feature_size
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        # compute the mean, standard of input x, and do normalization
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def subsequent_mask(size):
    "To create the mask matrix"
    attention_shape = (1, size, size)
    # create the upper triangular matrix using np.triu()
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')

    # change the type numpy to torch and reverse 0 and 1
    return torch.from_numpy(subsequent_mask) == 0


###########    Encoder    ############
class Encoder(nn.Module):
    """
    a stack made by 6 same layers
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # clone 6 same layers
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            return self.norm(x)


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        sublayer_out = sublayer(x)

        # The scheme of the original tutorial.
        # x_norm = self.norm(x + self.dropout(sublayer_out))

        # take x out of norm may converge faster, but I can make sure that it's right
        sublayer_out = self.dropout(sublayer_out)
        x_norm = x + self.norm(sublayer_out)
        return x_norm


class EncoderLayer(nn.Module):
    "Encoder have two layers: self-attention layer and FFN layer"
    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        # feed forward sublayer
        z = self.sublayer[1](x, self.feed_forward)
        return z


###########    Decoder    ############
class Decoder(nn.Module):
    """
    Same to Encoder, a stack made by 6 same layers
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: input of decoder
        :param memory: output of encoder
        :param src_mask: mask for source data
        :param tgt_mask: mask for target data
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attention, src_attention, feed_forward, dropout):
        "Decoder have 3 layers: self-attention layer, source-target attention layer, and FFN layer"
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # self-attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # source-target attention sublayer, in which q is from x, and k, v is from memory
        x = self.sublayer[1](x, lambda x: self.src_attention(x, m, m, src_mask))
        # ffn sublayer
        return self.sublayer[2](x, self.feed_forward)


##########  The Generator  #############
class Generator(nn.Module):
    "The Generator has 'Linear + softmax' step"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.softmax(self.proj(x), dim=-1)


##########     The standard Encoder-Decoder architecture    ######################
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src: input of encoder
        :param tgt: input of decoder
        memory: output of encoder
        res: output of decoder
        """
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        "The encode function"
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, memory, src_mask, tgt ,tgt_mask):
        "The decode function"
        tgt_embeds = self.tgt_embed(tgt)
        return self.decoder(tgt_embeds, memory, src_mask, tgt_mask)


# FULL MODEL
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model