import os
import sys
import time

import math
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from path import *
from data import *
from config import DETERRENTConfig
from model.DETERRENT import DETERRENT


def data_generator(datas, big_graphs, config, shuffle=True, k_hop=0):

    batch_size = config.batch_size
    data_size = len(datas)
    batch_num = math.ceil(data_size / batch_size)


    if shuffle:
        random.shuffle(datas)
    for batch_idx in range(batch_num):
        batch = datas[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        for data in batch:

            #--------------------------#
            #           graph          # 
            #--------------------------#

            # test entities & triples
            text_entities = data["text_mentioned_entities"]
            # 1. 1hop full triples
            test_triples = None
            season = big_graphs  # set of tuples
            entities_selected = get_k_hop_entities(season, text_entities, k_hop)
            after_triples = get_triples_by_entities(season, entities_selected)

            # get all possible triples
            test_triples = get_all_possible_triples(set(after_triples), set(entities_selected), set([i for i in range(config.relation_size)]))
            triple_head_indices, triple_relation_indices, triple_tail_indices = zip(*test_triples)

            # labels
            labels = [data["label"]]

            #--------------------------#
            #           adj            # 
            #--------------------------#

            # adjacent matrices
            adjacents = get_adjacents_selfloop(season, config.entity_size, config.relation_size)

            #--------------------------#
            #           text           # 
            #--------------------------#

            # text
            text_indices, text_length = padding_sequence(data["text_mentioned_tokens"], max_length=config.text_max_length, pad_idx=config.pad_idx, get_length=True)

            # generate input and output dict
            input_dict = { 
                "entity_indices": torch.arange(config.entity_size).cpu(),
                "text_indices": torch.unsqueeze(torch.LongTensor(text_indices), dim=0).cpu(),
                "text_lengths": torch.LongTensor([text_length]).cpu(),
                # "triple_head_indices": torch.LongTensor(triple_head_indices).cpu(),
                # "triple_relation_indices": torch.LongTensor(triple_relation_indices).cpu(),
                # "triple_tail_indices": torch.LongTensor(triple_tail_indices).cpu()
            }
            input_dict.update([("adjacent_%s" % i, adjacent.cpu()) for i, adjacent in enumerate(adjacents)])

            output_dict = {
                "score": torch.FloatTensor(labels).cpu()
            }

            yield input_dict, output_dict


def test():
    
    # init config
    print("init config ...")
    config = DETERRENTConfig()
    model_name = config.model_name
    # select best model
    max_f1, max_model = 0, ""
    for name in os.listdir(os.path.join(path_model_dir, model_name)):
        f1 = float(name.split("_")[-1])
        if f1 > max_f1:
            max_f1 = f1 
            max_model = name
    path_model = os.path.join(path_model_dir, model_name, max_model)

    # load data 
    print("load data ...")
    datas_test  = load_articles(path_test)
    big_graphs = gen_big_graphs(path_graph)
    print("test data size: ", len(datas_test))

    # gpu config
    print("set gpu and init model ...")
    print("load model from:", path_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    
    # test
    print("start testing ...")

    # model
    model = DETERRENT(config).cpu()
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()

    data_size = 0
    datas_test = [data for data in datas_test if data["text"]]
    acc_num, total_num = 0, 0
    label_acc_num, label_total_num = 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    epsilon = 1e-6

    with torch.no_grad():
        for input_dict, output_dict in data_generator(datas_test, big_graphs, config, shuffle=False,
                                                      k_hop=config.k_hop):
            scores = model(input_dict)

            acc_num += compute_acc(scores, output_dict["score"])
            total_num += len(output_dict["score"])
            TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
            FN += torch.sum(torch.sign((scores < 0.5).float() * output_dict["score"])).item()
            FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
            TN += torch.sum(torch.sign((scores < 0.5).float() * (1 - output_dict["score"]))).item()

    accuracy = acc_num / (total_num + epsilon)
    label_accuracy = label_acc_num / (label_total_num + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    print("accuracy: %d/%d=%.4f" % (acc_num, total_num, accuracy))
    print()


    

# ====================== main =========================

def main():
    test()

if __name__ == '__main__':
    main()