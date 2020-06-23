import os
import sys
import time


import csv
import math
import json
import pickle
import re
import copy
import random

from collections import Counter

import numpy as np
import torch


# ===================== load & save =========================

def load_json(file_path):
    ''' load json file '''
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
        return data

def dump_json(data, file_path):
    ''' save json file '''
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_pkl(path):
    ''' load pkl '''
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(data, path):
    ''' save pkl '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_str_lst(path):
    ''' load string list '''
    strs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            strs.append(line.strip())
    return strs

def load_str_dict(path, seperator="\t", reverse=False):
    ''' load string dict '''
    dictionary, reverse_dictionay = {}, {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                key, value = line.strip().split(seperator)
                dictionary[key] = int(value)
                reverse_dictionay[int(value)] = key
            except:
                pass
    if reverse:
        return dictionary, reverse_dictionay, len(dictionary)
    return dictionary, len(dictionary)

def dump_str_lst(lst, path):
    ''' save string list '''
    with open(path, "w", encoding="utf8") as f:
        for string in lst:
            f.write(string+'\n')

def load_articles(file_path):
    ''' load data '''
    datas = load_json(file_path)
    # for i in range(len(datas)):
    #     datas[i]["subgraph_before"] = [tuple(triple) for triple in datas[i]["subgraph_before"]]
    return datas

# =========== task specific transformation ===========

def gen_big_graphs(file_path):
    ''' generate big graphs by season '''
    big_graphs = load_json(file_path)
    return random.sample(big_graphs, 2000)


def get_k_hop_entities(triples, seeds, k=0):
    '''
        get k-hop neighbors of seeds from triples
        parameters:
            @triples: set((h,r,t), ...)
            @seeds: set(e1, e2, ...)
            @k: h-hop, int
    '''
    ret_triples, ret_entities = set(), set(seeds)
    for _ in range(k):
        new_seeds = set()
        for h, r, t in triples:
            if h in seeds or t in seeds:
                ret_triples.add((h, r, t))
                ret_entities.add(h)
                ret_entities.add(t)
                new_seeds.update([h, t])
        seeds = new_seeds
    return list(ret_entities)


def get_triples_by_entities(triples, entities):
    ''' return all triples that head & tail in entities '''
    return [(h, r, t) for h, r, t in triples if h in entities and t in entities]


def get_all_possible_triples(ref_triples, entities, relations, neg_label=0.):
    '''
        根据entities和relations生成全图，并给出对应标注
        参数:
            ref_triples: set((h,r,t), ...)
            entities:    set(e1, e2)
            relations:   set(r1, r2)
        返回:
            full_triples: [(h, r, t), ...]
            labels:       [1, 0, ...]
    '''

    full_triples = []
    for h in entities:
        for t in entities:
            if h == t:
                continue
            for r in relations:
                if (h, r, t) in ref_triples:
                    full_triples.append((h, r, t))
    return full_triples


def padding_sequence(indices, max_length, pad_idx, get_length=False):
    ''' '''
    length = len(indices) if len(indices) < max_length else max_length
    if len(indices) >= max_length:
        if get_length:
            return indices[:max_length], length
        else:
            return indices[:max_length]
    else:
        if get_length:
            return indices + [pad_idx] * (max_length - len(indices)), length
        else:
            return indices + [pad_idx] * (max_length - len(indices))

def get_adjacents_selfloop(triples, entity_size, relation_size):

    '''
        generate adjacent matrix for each relation type
        add new relation type <self-loop>
    '''

    # init edges
    edges_lst = [[] for _ in range(relation_size)]
    for h, r, t in triples:
        edges_lst[r].append((h, t))

    # count edge
    neighbor_counter = Counter()
    indices_pairs = []
    for edges in edges_lst:
        neighbors = [e[0] for e in edges]
        neighbor_counter.update(neighbors)
        edges.sort()
        if edges:
            row_indices, col_indices = zip(*edges)
        else:
            row_indices, col_indices = (), ()
        indices_pairs.append((row_indices, col_indices))
    # add identity (self-loop)
    neighbor_counter.update([i for i in range(entity_size)])
    identity_indices = [i for i in range(entity_size)]
    indices_pairs.append((identity_indices, identity_indices))

    # r-normalization
    adjacents = []
    for row_indices, col_indices in indices_pairs:
        if row_indices:
            weights = torch.FloatTensor([1. for i in range(len(row_indices))])
            indices = torch.LongTensor([row_indices, col_indices])
            adjacent = torch.sparse.FloatTensor(indices, weights, torch.Size([entity_size, entity_size]))
        else:
            adjacent = torch.sparse.FloatTensor(entity_size, entity_size)
        adjacents.append(adjacent)

    return adjacents

# ==================== metrics =======================

def my_pos_avg(y_pred, y_true):
    return torch.mean((y_true + 1.) * y_pred / 2.0)
def my_neg_avg(y_pred, y_true):
    return torch.mean((1. - y_true) * y_pred / 2.0)
def compute_acc(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true) < 0.5).item()
def F1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.

# ===================== main =========================

def main():
    pass

if __name__ == '__main__':
    main()