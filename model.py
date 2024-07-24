import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import random
from torch.autograd import Variable
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # print(self.pe[:, :x.size(1)].shape)
        # print(x.shape)
        x = x + self.pe[:, :x.size(1)]
        return x


# ################################# Cross attention #####################################

class _CrossAttention(nn.Module):
    def __init__(self, input_dim_query, input_dim_key_value, embed_dim, num_heads):
        super().__init__()
        assert (
                embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim_query, embed_dim)
        self.kv_proj = nn.Linear(input_dim_key_value, 2 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, y, mask=None, return_attention=False):
        batch_size, query_seq_length, _ = x.size()
        key_value_seq_length = y.size(1)

        q = self.q_proj(x)
        kv = self.kv_proj(y)

        q = q.reshape(batch_size, query_seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [Batch, Head, QuerySeqLen, HeadDim]
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(batch_size, key_value_seq_length, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [Batch, Head, KeySeqLen, HeadDim]
        v = v.reshape(batch_size, key_value_seq_length, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [Batch, Head, ValueSeqLen, HeadDim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attn_weights, v)

        # Merge heads
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, query_seq_length, -1)

        # Project back to embedding dimension
        o = self.o_proj(attended_values)

        if return_attention:
            return o, attn_weights
        else:
            return o


class _EncoderBlock_cross(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.cross_attn = _CrossAttention(input_dim, input_dim, input_dim, num_heads)

        # # Two-layer MLP
        # self.linear_net = nn.Sequential(
        #     nn.Linear(input_dim, dim_feedforward),
        #     nn.Dropout(dropout),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_feedforward, input_dim),
        # )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        # Attention part
        attn_out = self.cross_attn(x, y, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        return x


class _TransformerEncoder_cross(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderBlock_cross(**block_args) for _ in range(num_layers)])

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.cross_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

    def forward(self, x, y, mask=None):
        for l in self.layers:
            x = l(x, y, mask=mask)
        return x


class Cross_attention(nn.Module):
    def __init__(self,
                 input_dim=None,
                 model_dim=None,
                 num_layers=1,
                 num_heads=2,
                 input_dropout=0.2,
                 dropout=0.5):
        super(Cross_attention, self).__init__()

        # Transformer
        self.transformer_cross = _TransformerEncoder_cross(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, y, mask=None):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = self.transformer_cross(x, y, mask=mask)
        return x


# ################################# Self attention #####################################

class _MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
                embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class _EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = _MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        # self.linear_net = nn.Sequential(
        #     nn.Linear(input_dim, dim_feedforward),
        #     nn.Dropout(dropout),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_feedforward, input_dim),
        # )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class Self_attention(nn.Module):
    def __init__(self,
                 input_dim=None,
                 model_dim=None,
                 num_layers=1,
                 num_heads=2,
                 input_dropout=0.2,
                 dropout=0.5):
        super(Self_attention, self).__init__()

        # Transformer
        self.transformer = _TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = self.transformer(x, mask=mask)
        return x


# ################################# Video encoder #####################################

class Video_encoder_(nn.Module):

    def __init__(self,
                 input_dim=None,
                 model_dim=None,
                 num_layers=1,
                 num_heads=2,
                 input_dropout=0.2,
                 dropout=0.5):
        super(Video_encoder_, self).__init__()

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim),
        )

        # Positional encoding for sequences
        self.positional_encoding = _PositionalEncoding(d_model=model_dim)

        # Transformer
        self.self_attention = Self_attention(model_dim=model_dim)

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.LayerNorm(model_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = x.unsqueeze(0)
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.self_attention(x, mask=mask)
        out = self.output_net(x)

        # delete dimention of 'batch_size'
        out = out.squeeze(0)
        return out


# ################################# Audio encoder #####################################

class Audio_encoder_(nn.Module):

    def __init__(self,
                 input_dim=None,
                 model_dim=None,
                 num_layers=1,
                 num_heads=2,
                 input_dropout=0.2,
                 dropout=0.5):
        super(Audio_encoder_, self).__init__()

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim),
        )

        # Positional encoding for sequences
        self.positional_encoding = _PositionalEncoding(d_model=model_dim)

        # Transformer
        self.self_attention = Self_attention(model_dim=model_dim)

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.LayerNorm(model_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = x.unsqueeze(0)
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.self_attention(x, mask=mask)
        out = self.output_net(x)
        # delete dimention of 'batch_size'
        out = out.squeeze(0)

        return out


# ################################# Integrated encoder #####################################

class VA_encoder_self_cross_sigmoid(nn.Module):
    def __init__(self, input_dim=None, model_dim=None):
        super(VA_encoder_self_cross_sigmoid, self).__init__()
        self.video_encoder = Video_encoder_(input_dim=input_dim, model_dim=model_dim)  # SA
        self.audio_encoder = Audio_encoder_(input_dim=input_dim, model_dim=model_dim)  # SA

        self.cross_attention_video = Cross_attention(model_dim=model_dim)  # CA
        self.cross_attention_audio = Cross_attention(model_dim=model_dim)  # CA
        self.mu_linear = nn.Linear(model_dim, 1)  # Through W, output distribution μ
        self.mu_sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x, y):
        v = self.video_encoder(x)
        a = self.audio_encoder(y)

        v_a = self.cross_attention_video(v, a)
        a_v = self.cross_attention_audio(a, v)

        mu = self.mu_linear(v_a)
        mu = self.mu_sigmoid(mu)
        mu = mu.squeeze(0)

        v_a = v_a.squeeze(0)
        a_v = a_v.squeeze(0)
        return v_a, mu, a_v
