import os
import sys
import time

from path import *
from data import *

class DETERRENTConfig():

    def __init__(self):

        # basic
        self.model_name = "DETERRENT_model"
        self.graph_embedding_dim = 128
        self.hidden_dim = 128
        self.batch_size = 100
        self.epoch_num = 10
        self.report_step_num = 10
        self.dropout_rate = 0.5
        self.learning_rate = 1e-3
        self.min_learning_rate = 1e-4
        self.decay_factor = 0.3
        self.patience = 2

        # task specific
        self.text_max_length = 120
        self.pad_idx = 6691
        self.basis_num = 2
        self.use_text = True
        self.k_hop = 1

        # train
        self.gpu_id = "0"

        # vocab
        self.entity_path = path_entity2id
        self.relation_path = path_relation2id
        self.token_path = path_token2id

        # init
        self.init()


    def init(self):
        ''' additional configuration '''

        # vocab
        # self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
        # self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
        # self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)
        self.entity_size = 24806
        self.relation_size = 10
        self.token_size = 57300

        # extra adjacent matrix number
        self.add_adj_size = 1  # selfloop

