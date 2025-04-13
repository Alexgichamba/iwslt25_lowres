import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple
from e2e_st.model.speech_embedding import WhisperSpeechEmbedding, SpeechTransformerSpeechEmbedding
from e2e_st.utils.attention_masks import key_padding_mask, causal_mask

class PositionalEncoding(nn.Module):
    """
    Sinusoidal position encoding from:
    "Attention Is All You Need" by Vaswani et al.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, max_len):
        super().__init__()

        # initialize the positional encodings
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)

        # list all positions (0 to max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)

        # compute the division term for the sine and cosine functions
        # generates varying frequencies for positional encodings
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices

        # reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, T, d_model)
            offset: int (optional) to specify the starting position of the positional encodings, useful for kv_cache
        Returns:
            x: torch.Tensor of shape (N, T, d_model)
        """
        # add the positional encodings to the input tensor
        x = x + self.pe[:, offset:offset + x.size(1)]
        return x

class FeedForward(nn.Module, ABC):
    """
    Abstract base class for feed-forward networks in transformer architectures.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, T, d_model)
        Returns:
            x: torch.Tensor of shape (N, T, d_model)
        """
        pass

class FFNTransformer(FeedForward):
    """
    FeedForward Neural Network in the original transformer from:
        "Attention Is All You Need" by Vaswani et al.
        https://arxiv.org/abs/1706.03762
    Uses GELU instead of the original ReLU.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x

class FFNSwiGLU(FeedForward):
    """
    Feed Forward Swish-Gated Linear Unit (FFNSwiGLU) network from:
        "GLU Variants Improve Transformer" by Shazeer
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        
        d_ff_reduced = int(d_ff * (2/3))
        self.w = nn.Linear(d_model, d_ff_reduced)
        self.v = nn.Linear(d_model, d_ff_reduced)
        self.w2 = nn.Linear(d_ff_reduced, d_model)
        self.silu = nn.SiLU()  # Equivalent to Swish₁ when β=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_output = self.silu(self.w(x))  # Swish₁(x @ W)
        v_output = self.v(x)             # (x @ V)
        x = w_output * v_output          # Element-wise multiplication
        x = self.w2(x)                   # Final projection
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using torch SDPA from:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    KV caching support added.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True, use_sdpa: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads # floor division to ensure head_dim is an integer

        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout

        self.use_sdpa = use_sdpa

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                causal_attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> torch.Tensor:
        """
        Args:
            query: torch.Tensor of shape (N, T_q, d_model)
            key: torch.Tensor of shape (N, T_k, d_model)
            value: torch.Tensor of shape (N, T_k, d_model)
            key_padding_mask: Optional[torch.Tensor] of shape (N, T_k)
            causal_attn_mask: Optional[torch.Tensor] of shape (T_q, T_q)
            average_attn_weights: bool (return attention weights averaged over all heads)
        Returns:
            x: torch.Tensor of shape (N, T_q, d_model)
            attn_weights: torch.Tensor of shape (N, T_q, T_k) or (N, num_heads, T_q, T_k)
        """

        # project the query, key, and value
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        if self.training is False:
            self.dropout_p = 0.0

        # perform qkv_attention
        attn_output, attn_weights = self.qkv_attention(q=query,
                                                       k=key,
                                                       v=value,
                                                       key_padding_mask=key_padding_mask,
                                                       causal_attn_mask=causal_attn_mask,
                                                       dropout_p=self.dropout_p)

        # project the output
        attn_output = self.out(attn_output)

        if average_attn_weights and attn_weights is not None:
            # average attention weights over all heads
            attn_weights = attn_weights.mean(dim=1)

        return attn_output, attn_weights


        
    def qkv_attention(self,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      key_padding_mask: Optional[torch.Tensor] = None,
                      causal_attn_mask: Optional[torch.Tensor] = None,
                      dropout_p: float = 0.0
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the scaled dot-product attention.
        Args:
            q: torch.Tensor of shape (N, T_q, d_model)
            k: torch.Tensor of shape (N, T_k, d_model)
            v: torch.Tensor of shape (N, T_k, d_model)
            causal_attn_mask: Optional[torch.Tensor] of shape (T_q, T_q)
            key_padding_mask: Optional[torch.Tensor] of shape (N, T_k)
            dropout_p: float
        Returns:
            a: torch.Tensor of shape (N, T, d_model)
        """
        
        N, T_q, _ = q.size()
        _, T_k, _ = k.size()

        # Split the query, key, and value into multiple heads
        # (N, T, d_model) -> (N, T, num_heads, head_dim) -> (N, num_heads, T, head_dim)
        q = q.view(N, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # if both masks are provided, combine them. Note: True means to mask the value, False means to consider the value
        mask = None
        if causal_attn_mask is not None:
            assert T_q == T_k, "T_q and T_k must be equal for causal self attention"
            # expand the causal mask to all heads
            causal_attn_mask = causal_attn_mask.unsqueeze(0).unsqueeze(1).expand(N, self.num_heads, T_q, T_k)
            mask = causal_attn_mask
            if key_padding_mask is not None:
                # expand the key padding mask to all heads
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(N, self.num_heads, T_q, T_k)
                # combine the masks
                mask = causal_attn_mask | key_padding_mask
        elif key_padding_mask is not None:
            # expand the key padding mask to all heads
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(N, self.num_heads, T_q, T_k)
            mask = key_padding_mask

        attn_weights = None
        # Compute MHA with the optimized SDPA implementation
        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(query=q,
                                                key=k,
                                                value=v,
                                                attn_mask=mask,
                                                is_causal=False,
                                                dropout_p=dropout_p)
        else: # Compute MHA with the slower implementation
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale # (N, num_heads, T_q, T_k)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = attn_weights @ v # (N, num_heads, T_q, head_dim)
        # Combine the heads
        # (N, num_heads, T, head_dim) -> (N, T, num_heads, head_dim) -> (N, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, T_q, self.d_model)

        return attn_output, attn_weights


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer from:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Dong et al.
        https://ieeexplore.ieee.org/document/8462506
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 attn_dropout: float,
                 ff: Literal["transformer", "swiglu"],
                 bias: bool = True,
                 use_sdpa: bool = True):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=attn_dropout, bias=bias, use_sdpa=use_sdpa)
        if ff == "transformer":
            self.ff = FFNTransformer(d_model, d_ff)
        elif ff == "swiglu":
            self.ff = FFNSwiGLU(d_model, d_ff)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.residual_dropout1 = nn.Dropout(attn_dropout)
        self.residual_dropout2 = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, T, d_model)
            (Optional) pad_mask: torch.Tensor of shape (N, T)
        Returns:
            x: torch.Tensor of shape (N, T, d_model)
            attn_weights: torch.Tensor of shape (N, T, T)
        """

        # residual 1
        residual = x
        # pre normalization
        x = self.pre_norm(x)
        # self-attention
        x, attn_weights = self.self_attn(query=x,
                                         key=x,
                                         value=x,
                                         key_padding_mask=pad_mask,
                                         average_attn_weights=True)
        # residual connection
        x = self.residual_dropout1(x) + residual
        # post normalization
        x = self.post_norm(x)
        # residual 2
        residual = x
        # feed-forward network
        x = self.ff(x)
        # residual connection with dropout
        x = self.residual_dropout2(x) + residual
        return x, attn_weights

class Encoder(nn.Module):
    """
    Transformer Encoder from:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Dong et al.
        https://ieeexplore.ieee.org/document/8462506
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 n_layers: int,
                 attn_dropout: float,
                 in_channels: Optional[int],
                 speech_embedding_type: Literal["whisper", "speech_transformer"],
                 ff_type: Literal["transformer", "swiglu"],
                 pe_max_len: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 use_sdpa: bool = True):
        super().__init__()
        
        if speech_embedding_type == "whisper":
            self.speech_embedding = WhisperSpeechEmbedding(in_channels=in_channels, out_channels=d_model, dtype=dtype, device=device)
        elif speech_embedding_type == "speech_transformer":
            self.speech_embedding = SpeechTransformerSpeechEmbedding(out_channels=d_model, dtype=dtype, device=device)
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=pe_max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                                n_heads=n_heads,
                                                                d_ff=d_ff,
                                                                attn_dropout=attn_dropout,
                                                                ff=ff_type,
                                                                use_sdpa=use_sdpa) for _ in range(n_layers)])

        self.post_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, T, d_model)
            x_lengths (Optional): torch.Tensor of shape (N,)
        Returns:
            x: torch.Tensor of shape (N, T, d_model) (encoder output)
            x_lengths: torch.Tensor of shape (N,) (encoder output lengths)
            attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
        """
        
        # speech embedding
        x, x_lengths = self.speech_embedding(x, x_lengths)
        # permute to (N, T, d_model)
        x = x.permute(0, 2, 1)
        # positional encoding
        x = self.positional_encoding(x)
        # generate padding mask
        pad_mask = key_padding_mask(x, x_lengths).to(self.device)
        # attention weights will be stores in a dictionary where the key is the layer number
        attn_weights = {}
        # iterate over each encoder layer
        for i, encoder_layer in enumerate(self.encoder_layers):
            x, attn_weights[i] = encoder_layer(x, pad_mask)
        # post normalization
        x = self.post_norm(x)
        return x, x_lengths, attn_weights

class CTCHead(nn.Module):
    """
    Connectionist Temporal Classification (CTC) Head for speech recognition.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, T, d_model)
        Returns:
            x: torch.Tensor of shape (N, T, vocab_size)
        """
        return self.linear(x)

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer from:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Dong et al.
        https://ieeexplore.ieee.org/document/8462506
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 attn_dropout: float,
                 ff: Literal["transformer", "swiglu"],
                 bias: bool = True,
                 use_sdpa: bool = True):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=attn_dropout, bias=bias, use_sdpa=use_sdpa)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=attn_dropout, bias=bias, use_sdpa=use_sdpa)
        if ff == "transformer":
            self.ff = FFNTransformer(d_model, d_ff)
        elif ff == "swiglu":
            self.ff = FFNSwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.residual_dropout1 = nn.Dropout(attn_dropout)
        self.residual_dropout2 = nn.Dropout(attn_dropout)
        self.residual_dropout3 = nn.Dropout(attn_dropout)

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                causal_attn_mask: torch.Tensor,
                enc_pad_mask: Optional[torch.Tensor] = None,
                dec_pad_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: torch.Tensor of shape (N, T_q, d_model)
            enc_output: torch.Tensor of shape (N, T_k, d_model)
            causal_attn_mask: torch.Tensor of shape (T_q, T_q)
            (Optional) enc_pad_mask: torch.Tensor of shape (N, T_k)
            (Optional) dec_pad_mask: torch.Tensor of shape (N, T)
            average_attn_weights: bool (return attention weights averaged over all heads)
        Returns:
            x: torch.Tensor of shape (N, T, d_model)
            self_attn_weights: torch.Tensor of shape (N, T, T)
            cross_attn_weights: torch.Tensor of shape (N, T, T)
        """

        # residual 1
        residual = x
        # pre normalization
        x = self.norm1(x)
        # causal self-attention
        x, self_attn_weights = self.self_attn(query=x,
                                              key=x,
                                              value=x,
                                              key_padding_mask=dec_pad_mask,
                                              causal_attn_mask=causal_attn_mask,
                                              average_attn_weights=average_attn_weights
                                            )
        # residual connection
        x = self.residual_dropout1(x) + residual
        # normalization
        x = self.norm2(x)
        # residual 2
        residual = x
        # cross-attention
        x, cross_attn_weights = self.cross_attn(query=x,
                                                key=enc_output,
                                                value=enc_output,
                                                key_padding_mask=enc_pad_mask,
                                                causal_attn_mask=None,
                                                average_attn_weights=average_attn_weights
                                                )

        # residual connection
        x = self.residual_dropout2(x) + residual
        # normalization
        x = self.norm3(x)
        # residual 3
        residual = x
        # feed-forward network
        x = self.ff(x)
        # residual connection with dropout
        x = self.residual_dropout3(x) + residual

        return x, self_attn_weights, cross_attn_weights

class Decoder(nn.Module):
    """
    Transformer Decoder from:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Dong et al.
        https://ieeexplore.ieee.org/document/8462506
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 n_layers: int,
                 attn_dropout: float,
                 vocab_size: int,
                 ff: Literal["transformer", "swiglu"],
                 pe_max_len: int,
                 padding_idx: int,
                 device: torch.device,
                 use_sdpa: bool = True):
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                                n_heads=n_heads,
                                                                d_ff=d_ff,
                                                                attn_dropout=attn_dropout,
                                                                ff=ff, use_sdpa=use_sdpa)
                                                                for _ in range(n_layers)])

        self.post_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                enc_output_lengths: Optional[torch.Tensor] = None,
                padding_idx: Optional[int] = 3,
                average_attn_weights: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            x: torch.Tensor of shape (N, T_q, d_model)
            enc_output: torch.Tensor of shape (N, T_k, d_model)
            enc_output_lengths (Optional): torch.Tensor of shape (N,)
            padding_idx: int

        Returns:
            x: torch.Tensor of shape (N, T_q, d_model) (logits)
            self_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
            cross_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
        """

        # text embedding
        x = self.text_embedding(x)
        # positional encoding
        x = self.positional_encoding(x, offset=0)
        # generate padding mask
        dec_pad_mask = key_padding_mask(x, pad_idx=padding_idx).to(self.device)
        enc_pad_mask = key_padding_mask(enc_output, enc_output_lengths).to(self.device)
        # generate causal attention mask
        causal_attn_mask = causal_mask(x).to(self.device)
        # attention weights will be stored in a dictionary where the key is the layer number
        self_attn_weights = {}
        cross_attn_weights = {}
        # iterate over each decoder layer
        for i, decoder_layer in enumerate(self.decoder_layers):
            x, self_attn_weights[i], cross_attn_weights[i] = decoder_layer(x=x,
                                                                           enc_output=enc_output,
                                                                           causal_attn_mask=causal_attn_mask,
                                                                           enc_pad_mask=enc_pad_mask,
                                                                           dec_pad_mask=dec_pad_mask,
                                                                           average_attn_weights=average_attn_weights
                                                                            )
        # post normalization            
        x = self.post_norm(x)
        # final projection to logits (weight tying is used, so the same embedding matrix is used for input and output)
        x = F.linear(x, self.text_embedding.weight)
        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    Transformer model for end-to-end speech recognition.

    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 n_enc_layers: int,
                 n_dec_layers: int,
                 enc_attn_dropout: float,
                 dec_attn_dropout: float,
                 vocab_size: int,
                 in_channels: Optional[int],
                 speech_embedding_type: Literal["whisper", "speech_transformer"],
                 ff_type: Literal["transformer", "swiglu"],
                 pe_max_len: int,
                 padding_idx: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 average_attn_weights: bool = True,
                 use_sdpa: bool = True):
        super().__init__()

        self.encoder = Encoder(d_model=d_model,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               n_layers=n_enc_layers,
                               attn_dropout=enc_attn_dropout,
                               in_channels=in_channels,
                               speech_embedding_type=speech_embedding_type,
                               ff_type=ff_type,
                               pe_max_len=pe_max_len,
                               device=device,
                               dtype=dtype,
                               use_sdpa=use_sdpa)

        self.decoder = Decoder(d_model=d_model,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               n_layers=n_dec_layers,
                               attn_dropout=dec_attn_dropout,
                               vocab_size=vocab_size,
                               ff=ff_type,
                               pe_max_len=pe_max_len,
                               padding_idx=padding_idx,
                               device=device,
                               use_sdpa=use_sdpa)

        self.ctc_head = CTCHead(d_model=d_model, vocab_size=vocab_size)
        self.padding_idx = padding_idx
        self.average_attn_weights = average_attn_weights
        self.device = device

    def forward(self,
                speech_input: torch.Tensor,
                text_input: torch.Tensor,
                speech_lengths: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Args:
            speech_input: torch.Tensor of shape (N, n_mels, T_speech)
            text_input: torch.Tensor of shape (N, T_q)
            speech_lengths: Optional[torch.Tensor] of shape (N,) (lengths of the speech input)
        Returns:
            logits: torch.Tensor of shape (N, T, vocab_size)
            ctc_logits: torch.Tensor of shape (N, T, vocab_size)
            enc_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
            dec_self_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
            dec_cross_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T, T)
        """
        # encoder
        enc_output, enc_output_lengths, enc_attn_weights = self.embed_speech(speech_input, speech_lengths)
        # decoder
        dec_logits, dec_self_attn_weights, dec_cross_attn_weights = self.decode(x=text_input,
                                                                                     enc_output=enc_output,
                                                                                     enc_output_lengths=enc_output_lengths,
                                                                                     padding_idx=self.padding_idx,
                                                                                     average_attn_weights=self.average_attn_weights
                                                                                    )
        # CTC head
        ctc_logits = self.compute_ctc_logits(enc_output)
        return dec_logits, ctc_logits, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights

    def embed_speech(self,
                    speech_input: torch.Tensor,
                    speech_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speech_input: torch.Tensor of shape (N, n_mels, T_speech)
            speech_lengths: Optional[torch.Tensor] of shape (N,) (lengths of the speech input)
        Returns:
            enc_output: torch.Tensor of shape (N, T_k, d_model)
            enc_output_lengths: torch.Tensor of shape (N,)
        """
        enc_output, enc_output_lengths, enc_attn_weights = self.encoder(speech_input, speech_lengths)
        return enc_output, enc_output_lengths, enc_attn_weights
    
    def compute_ctc_logits(self,
                          enc_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_output: torch.Tensor of shape (N, T_k, d_model)
        Returns:
            ctc_logits: torch.Tensor of shape (N, T, vocab_size)
        """
        ctc_logits = self.ctc_head(enc_output)
        return ctc_logits
    
    def decode(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                enc_output_lengths: Optional[torch.Tensor] = None,
                padding_idx: Optional[int] = 3,
                average_attn_weights: bool = True
                ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Args:
            x: torch.Tensor of shape (N, T_q)
            enc_output: torch.Tensor of shape (N, T_k, d_model)
            enc_output_lengths (Optional): torch.Tensor of shape (N,)
            padding_idx: int
        Returns:
            dec_logits: torch.Tensor of shape (N, T_q, vocab_size)
            self_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T_q, T_q)
            cross_attn_weights: Dict[int, torch.Tensor] with keys as layer numbers and values as torch.Tensor of shape (N, T_q, T_k)
        """
        dec_logits, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(x=x,
                                                                                     enc_output=enc_output,
                                                                                     enc_output_lengths=enc_output_lengths,
                                                                                     padding_idx=padding_idx,
                                                                                     average_attn_weights=average_attn_weights
                                                                                    )
        return dec_logits, dec_self_attn_weights, dec_cross_attn_weights