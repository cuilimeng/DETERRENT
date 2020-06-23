
import os
import sys



# ============  model ==========

path_model_dir = os.path.join("", "model")

# ============ data ============

path_data_dir = os.path.join("", "data")

# vocab
path_entity2id = os.path.join(path_data_dir, "entity2id.txt")
path_relation2id = os.path.join(path_data_dir, "relation2id.txt")
path_token2id = os.path.join(path_data_dir, "token2id.txt")

# data
path_train = os.path.join(path_data_dir, "test.json")
path_valid = os.path.join(path_data_dir, "test.json")
path_test  = os.path.join(path_data_dir, "test.json")
path_graph  = os.path.join(path_data_dir, "output.json")