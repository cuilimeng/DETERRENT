import os
import sys
import time

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

            # all the triples
            season = big_graphs  # set of tuples

            # entities & triples
            text_entities = data["text_mentioned_entities"][:120] # list of numbers
            entities_selected = get_k_hop_entities(season, text_entities, k_hop) # list of numbers
            after_triples = get_triples_by_entities(season, entities_selected) # list of tuples

            # get all possible triples
            triples = get_all_possible_triples(set(after_triples), set(entities_selected), set([i for i in range(config.relation_size)])) # triples: list of tuples; labels: list of float numbers
            # shuffle
            # triples_labels = list(zip(triples, labels))
            # random.shuffle(triples_labels)
            # triples, labels = zip(*triples_labels)

            triple_head_indices, triple_relation_indices, triple_tail_indices = zip(*triples)

            #labels
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
                "triple_head_indices": torch.LongTensor(triple_head_indices).cpu(),
                "triple_relation_indices": torch.LongTensor(triple_relation_indices).cpu(),
                "triple_tail_indices": torch.LongTensor(triple_tail_indices).cpu()
            }
            input_dict.update([("adjacent_%s" % i, adjacent.cpu()) for i, adjacent in enumerate(adjacents)])

            output_dict = {
                "score": torch.FloatTensor(labels).cpu()
            }
            yield input_dict, output_dict


def train():

    # init config
    print("init config ...")
    config = DETERRENTConfig()

    # create dir
    model_name = config.model_name
    path_model = os.path.join(path_model_dir, model_name)
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    print("model name:", model_name)

    # load data 
    print("load data ...")
    datas_train = load_articles(path_train)
    random.shuffle(datas_train)
    TRAIN_SIZE = int(len(datas_train) * 0.8)
    datas_dev   = datas_train[TRAIN_SIZE:]
    datas_train = datas_train[:TRAIN_SIZE]
    big_graphs  = gen_big_graphs(path_graph)
    train_size  = len(datas_train)
    print("train data size: ", train_size)

    # gpu config
    print("set gpu and init model ...")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    # model & load
    model = DETERRENT(config).cpu()

    # train
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.patience, factor=config.decay_factor, verbose=True, min_lr=config.min_learning_rate)
    max_f1 = 0
    epsilon = 1e-6
    last_path = ""
    for epoch in range(config.epoch_num):
        # train
        model.train()
        print("start epoch %d:" % epoch)
        print("======")
        train_loss = 0
        pos_avg, neg_avg = 0, 0
        acc_num, total_num = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        step = 0
        time_start = time.time()
        for input_dict, output_dict in data_generator(datas_train, big_graphs, config, k_hop=config.k_hop):
            model.zero_grad()
            scores, brp_loss = model(input_dict)
            loss = loss_func(scores, output_dict["score"])
            # metrics
            pos_avg += my_pos_avg(scores, output_dict["score"])
            neg_avg += my_neg_avg(scores, output_dict["score"])
            acc_num += compute_acc(scores, output_dict["score"])
            total_num += len(output_dict["score"])

            # label accuracy
            TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
            FN += torch.sum(torch.sign((scores <  0.5).float() * output_dict["score"])).item()
            FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
            TN += torch.sum(torch.sign((scores <  0.5).float() * (1 - output_dict["score"]))).item()
            precision = TP / (TP + FP + epsilon)
            recall    = TP / (TP + FN + epsilon)
            f1        = 2 * precision * recall / (precision + recall + epsilon)

            brp_loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            train_loss += loss
            step += 1

            # Accelerate
            if step == len(datas_train) // config.batch_size:
                break
            if step % config.report_step_num == 0:
                time_end = time.time()
                print("epoch %d, step: %d,  %%%.2f/%%100, train_loss: %.4f, train_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f,  time_spent: %.2f min" %
                      (epoch, step, step/train_size*100, train_loss/step, acc_num/total_num, precision, recall, f1, (time_end-time_start)/60))
                

        print("*****")
        # dev
        model.eval()
        dev_loss = 0
        step = 0
        acc_num, total_num = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        with torch.no_grad():
            for input_dict, output_dict in data_generator(datas_dev, big_graphs, config, k_hop=config.k_hop):
                scores, brp_loss = model(input_dict)
                loss = loss_func(scores, output_dict["score"])
                acc_num += compute_acc(scores, output_dict["score"])

                # label accuracy
                total_num += len(output_dict["score"])
                TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
                FN += torch.sum(torch.sign((scores <  0.5).float() * output_dict["score"])).item()
                FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
                TN += torch.sum(torch.sign((scores <  0.5).float() * (1 - output_dict["score"]))).item()
                dev_loss += loss
                step += 1

                # Accelerate
                if step == len(datas_dev) // config.batch_size:
                    break
        precision = TP / (TP + FP + epsilon)
        recall    = TP / (TP + FN + epsilon)
        f1        = 2 * precision * recall / (precision + recall + epsilon)
        print("epoch %d dev, dev_loss: %.4f, dev_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" %
              (epoch, dev_loss/step, acc_num/total_num, precision, recall, f1))

        # save
        if f1 > max_f1:
            if last_path:
                print("remove %s" % (last_path))
                os.remove(last_path)
            ckpt_name = "epoch_%d_loss_%.4f_f1_%.4f" % (epoch, dev_loss/step, f1)
            save_path = os.path.join(path_model, ckpt_name)
            last_path = save_path
            print("f1 from %.4f -> %.4f, saving model to %s" % (max_f1, f1, save_path))
            torch.save(model.state_dict(), save_path)
            max_f1 = f1
        # lr scheduler
        scheduler.step(f1)

        print()

    

# ====================== main =========================

def main():
    train()

if __name__ == '__main__':
    main()