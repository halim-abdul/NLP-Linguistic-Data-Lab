import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{config.hidden_size} vs {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        batch_size, seq_len = x.shape[:2]

        projected = linear_layer(x)
        projected = projected.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )

        return projected.transpose(1, 2)

    def attention(self, key, query, value, attention_mask):

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask

        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()

        batch_size, seq_len = context.shape[:2]
        context = context.view(batch_size, seq_len, self.all_head_size)

        return context

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transform(hidden_states, self.query)
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)

        return self.attention(
            key=key_layer,
            query=query_layer,
            value=value_layer,
            attention_mask=attention_mask,
        )


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = BertSelfAttention(config)

        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu

        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):

        output = dense_layer(output)
        output = dropout(output)
        output = ln_layer(input + output)

        return output


    def forward(self, hidden_states, attention_mask):
        attention_output = self.self_attention(hidden_states, attention_mask)

        attention_output = self.add_norm(
            input=hidden_states,
            output=attention_output,
            dense_layer=self.attention_dense,
            dropout=self.attention_dropout,
            ln_layer=self.attention_layer_norm,
        )

        intermediate_output = self.interm_dense(attention_output)
        intermediate_output = self.interm_af(intermediate_output)

        layer_output = self.add_norm(
            input=attention_output,
            output=intermediate_output,
            dense_layer=self.out_dense,
            dropout=self.out_dropout,
            ln_layer=self.out_layer_norm,
        )
        return layer_output


class BertModel(BertPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.word_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )

        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Input sequence length {seq_length} is larger than "
                f"max_position_embeddings={self.config.max_position_embeddings}."
            )

        inputs_embeds = self.word_embedding(input_ids)

        position_ids = self.position_ids[:, :seq_length]
        position_ids = position_ids.expand(input_shape)
        position_embeds = self.pos_embedding(position_ids)

        token_type_ids = torch.zeros(
            input_shape,
            dtype=torch.long,
            device=input_ids.device,
        )
        token_type_embeds = self.tk_type_embedding(token_type_ids)

        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)

        return embeddings
        return embeddings

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask = get_extended_attention_mask(
            attention_mask,
            self.dtype,
        )

        for layer_module in self.bert_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)

        sequence_output = self.encode(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )

        cls_output = sequence_output[:, 0]
        pooled_output = self.pooler_dense(cls_output)
        pooled_output = self.pooler_af(pooled_output)

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
        }