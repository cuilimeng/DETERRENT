
import os
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.layers.TextRelationalGraphAttention import TextRelationalGraphAttention
from model.layers.GRUEncoder import GRUEncoder

class DETERRENT(nn.Module):
    def __init__(self, config):
        super(DETERRENT, self).__init__()
        self.config = config

        self.token_embedding    = nn.Embedding(config.token_size, config.graph_embedding_dim)
        self.relation_embedding = nn.Embedding(config.relation_size, config.graph_embedding_dim) 

        # text
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)

        # encoder
        self.r_gat = TextRelationalGraphAttention(self.config.graph_embedding_dim, self.config.hidden_dim, self.config.hidden_dim,
                                                   self.config.relation_size+self.config.add_adj_size, basis_num=self.config.basis_num,
                                                   activation="relu", use_text=config.use_text)

        self.drop  = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(config.hidden_dim, config.graph_embedding_dim)

        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(config.hidden_dim * 2, 1)
        self.dense3 = nn.Linear(config.hidden_dim, config.hidden_dim * 2)

        # lstm layer
        self.lstm = nn.LSTM(self.config.graph_embedding_dim,
                            self.config.hidden_dim,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.2,
                            batch_first=True)

    def forward(self, input_dict):

        # unpack inputs
        entity_indices = input_dict["entity_indices"]
        text_indices = input_dict["text_indices"]
        text_lengths = input_dict["text_lengths"]
        triple_head_indices = input_dict["triple_head_indices"]
        triple_relation_indices = input_dict["triple_relation_indices"]
        triple_tail_indices = input_dict["triple_tail_indices"]

        adjacents = [input_dict["adjacent_%d" % i] for i in range(self.config.relation_size + self.config.add_adj_size) if ("adjacent_%d" % i) in input_dict]

        # embedding and encoding
        entity_embeddings = self.token_embedding(entity_indices)
        text_embeddings = self.token_embedding(text_indices)
        text_encodings = self.gru(text_embeddings, text_lengths)

        # shortcuts labeling
        relation_num = len(adjacents)
        adj_to_use = [i for i in range(2)]

        # R-GAT fusion
        fusioned_entity_embeddings = self.r_gat([entity_embeddings, text_encodings, adj_to_use] + adjacents)
        fusioned_entity_embeddings = self.dense(fusioned_entity_embeddings)

        # DistMult decode
        triple_heads = F.embedding(triple_head_indices, fusioned_entity_embeddings)
        triple_tails = F.embedding(triple_tail_indices, fusioned_entity_embeddings)
        triple_relations = self.relation_embedding(triple_relation_indices)

        # score
        mask = [1 if i<5 else -1 for i in triple_relation_indices]
        mask = torch.tensor(mask, dtype=torch.float)
        score = triple_heads * triple_relations * triple_tails # 2600*128
        score = torch.sum(score, dim=-1) # 2600
        score = torch.dot(score, mask)
        score = torch.sigmoid(score) # 2600
        score = - 0.001 * torch.log(score)

        # Text encoder
        packed_embedded = nn.utils.rnn.pack_padded_sequence(text_embeddings, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        W = nn.Parameter(torch.randn(1, fusioned_entity_embeddings.shape[0]))
        transformed_entity_embeddings = torch.mm(W, fusioned_entity_embeddings)
        # print(transformed_entity_embeddings.shape)
        # print(hidden.shape)

        gamma = 0.01
        knowledge_guided_hidden = (1-gamma) * hidden + gamma * self.dense3(transformed_entity_embeddings)

        predicted = self.dense2(knowledge_guided_hidden)
        predicted = torch.sigmoid(predicted)
        predicted = torch.squeeze(predicted, dim=0)
        return predicted, score
