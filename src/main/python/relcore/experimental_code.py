# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2023 - Mtumbuka F.                                                       #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2023.1"
__date__ = "08 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import os
import json
import torch
from transformers import AutoTokenizer, BertModel, BertForMaskedLM
from pytorch_metric_learning import losses

# if __name__ == '__main__':
# x = []
# y = []
# for i in range(1, 21):
#     if i % 5 != 0:
#         y.append(i)
#     else:
#         y.append(i)
#         x.append(y)
#         y = []
#
# b_input = torch.LongTensor(x)
# input_index = torch.LongTensor(
#     [
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 1]
#     ]
# )
# # # Check shape
# # print("Inputs: {} , ent_pos: {}".format(b_input.shape, input_index.shape))
# #
# # # Reshape tensors
# # inputs = b_input.view(-1)
# # ent_pos = input_index.view(-1)
# #
# # print("RESHAPE** Inputs: {} , ent_pos: {}".format(inputs.shape, ent_pos.shape))
# # inputs = torch.where(ent_pos == 1, inputs, ent_pos)
# #
# # # Get nonzero elements
# # indices = torch.nonzero(inputs)
# # print(inputs)
# # final = torch.gather(inputs, 0, indices.view(-1))
# # print(final)
# # print()
# # """Bert"""
# # b_input = torch.LongTensor(
# #     [
# #         [1, 2, 3, 4, 5],
# #         [1, 2, 3, 4, 5]
# #     ]
# # )
# # input_index = torch.LongTensor(
# #     [
# #         [0, 1, 0, 0, 0],
# #         [0, 0, 1, 0, 0]
# #     ]
# # )
# # bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
# # # bert.resize_token_embeddings(31600)
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# #
# # labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# # inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")["input_ids"]
# #
# # labels = torch.where(inputs == tokenizer.mask_token_id, labels, -100)
# # outputs = bert(input_ids=inputs, labels=labels)
# # print(outputs.loss)
# # print()
# # print(outputs.logits.shape)
# # exit()
# bert = BertModel.from_pretrained("bert-base-uncased")
# output_bert = bert(b_input)
# output = output_bert.last_hidden_state
# print(output.shape, input_index.shape)
# output = output.view(-1, 768)
# input_index = input_index.view(-1)
# print(output.shape, input_index.shape)
# indices = torch.nonzero(input_index)
# print("***")
# print(output)
# print()
# specific_ents = output[indices.view(-1)]
# print(specific_ents.shape)
# print(specific_ents)
# print()
# specific_ents = specific_ents.unsqueeze(1)
# print(specific_ents.shape)
# print(specific_ents)
# print()

# """ Contrastive loss"""
# cp_loss = losses.NTXentLoss(temperature=0.07)
#
# embedding = torch.nn.Embedding(10, 3)
#
# input_ = torch.LongTensor(
#     [
#         [1, 2, 3, 4],
#         [5, 6, 7, 8]
#     ]
# )
#
# mask = torch.LongTensor(
#     [
#         [1, 0, 0, 0],
#         [0, 0, 0, 1]
#     ]
# )
# labels = torch.LongTensor(
#     [
#         [0],
#         [0]
#     ]
# )
#
# output = embedding(input_)
# print("Input: {}, Labels: {}".format(input_.shape, labels.shape))
# print("Output: {}".format(output.shape))
#
# print()
# print(output)
# print()
# output = output.view(-1, output.shape[2])
# mask = mask.view(-1)
# print("After reshaping*****")
# print("Output: {}, Mask: {}".format(output.shape, mask.shape))
#
# mask_indices = torch.nonzero(mask)
# print("Nonzero: {}".format(mask_indices.shape))
# print(mask_indices)
# print()
#
# print("*******")
# output = output[mask_indices.view(-1)]
# print("Output shape: {}".format(output.shape))
# print(output)
#
# print("#### Loss ~~~~")
# loss = cp_loss(output, labels.view(-1))
# print(loss)

"""
ACE DATA STATS
    import collections.abc as collections
    import json
    import os
    partitions = ["dev", "test", "train"]
    files = ["en-dev.json", "en-test.json", "en-train.json"]
    root_data_dir = "/home/frankmtumbuka/Projects/EnCore/data/ace_2005"

    rs = []
    for partition, file in zip(partitions, files):
        relations = []
        data_path = os.path.join(root_data_dir, partition, file)
        entity_bio = os.path.join(root_data_dir, partition, "entity_BIO.json")
        tokens = os.path.join(root_data_dir, partition, "token.json")
        data = json.load(open(data_path, "r"))
        bio_data = json.load(open(entity_bio, "r"))
        tokens_data = json.load(open(tokens, "r"))
        print("====== {} ======".format(partition))
        print("Num of samples: {}".format(len(data)))
        print("Num of BIO... : {}".format(len(bio_data)))
        print("Tokens data ...: {}".format(len(tokens_data)))
        print()
        for sample in data:
            if len(sample["golden-relation-mentions"]) > 0:
                for relation in sample["golden-relation-mentions"]:
                    relations.append(relation["relation-type"])
        r_counter = collections.Counter(relations)
        r = set(relations)
        rs += relations
        print("Num of distinct rels: {}".format(len(r)))
        print("Relations: {}".format(r))
        print("Counter: {}".format(r_counter))
        print()
        print()
        bio_tags_fine_grained = []
        bio_tags_course_grained = []
        for i in bio_data:
            for x in i:
                bio_tags_fine_grained += x

        fine_grained_bio_counter = collections.Counter(bio_tags_fine_grained)
        fine_grained_bio_set = set(bio_tags_fine_grained)
        print("Ace Fine Grained NER BIO Counter: {}".format(fine_grained_bio_counter))
        print("ACE fine grained NER BIO set: {}".format(fine_grained_bio_set))
        print()
        print()
        for tag in bio_tags_fine_grained:
            if tag == "O":
                bio_tags_course_grained.append(tag)
            else:
                tags = tag.split(":")
                bio_tags_course_grained.append(tags[0])

        coarse_grained_bio_counter = collections.Counter(bio_tags_course_grained)
        coarse_grained_bio_set = set(bio_tags_course_grained)
        print("Ace Coarse Grained NER BIO Counter: {}".format(coarse_grained_bio_counter))
        print("ACE Coarse grained NER BIO set: {}".format(coarse_grained_bio_set))
        print()
        print()

    overall_counter = collections.Counter(rs)
    overall_distinct_relations = set(rs)
    print("Distinct relations: {}".format(len(overall_distinct_relations)))
    print("Rs: {}".format(overall_distinct_relations))
    print("Rs counter: {}".format(overall_counter))
"""

# arg_1_b = "B-Arg-1"
# arg_1_i = "I-Arg-1"
# arg_2_b = "B-Arg-2"
# arg_2_i = "I-Arg-2"
# x = ["Arg-1", "Arg-1", "Arg-1", "O", "O", "O", "Arg-2", "Arg-2"]
# new_x = ["O"] * len(x)
#
# for idx, tag in enumerate(x):
#     if tag == "Arg-1" and (idx - 1) >= 0 and x[idx - 1] != "Arg-1":
#         new_x[idx] = arg_1_b
#     elif tag == "Arg-1" and (idx - 1) >= 0 and x[idx - 1] == "Arg-1":
#         new_x[idx] = arg_1_i
#     elif tag == "Arg-2" and (idx - 1) >= 0 and x[idx - 1] != "Arg-2":
#         new_x[idx] = arg_2_b
#     elif tag == "Arg-2" and (idx - 1) >= 0 and x[idx - 1] == "Arg-2":
#         new_x[idx] = arg_2_i
#     elif tag == "Arg-1" and idx == 0:
#         new_x[idx] = arg_1_b
#     elif tag == "Arg-2" and idx == 0:
#         new_x[idx] = arg_2_b
#     else:
#         pass
#
# print(x)
# print(new_x)


# TACRED
path = "/home/frankmtumbuka/Projects/RelCore/data/"
dsets = ["tacred", "retacred", "tacrev"]
files = ["dev.json", "test.json", "train.json"]
import collections.abc as collections
if __name__ == '__main__':
    # Analysis
    tacred_training = []
    tacred_test = []
    tacred_dev = []

    t_train = open(os.path.join(path, dsets[1], "train.json"))
    t_test = open(os.path.join(path, dsets[1], "test.json"))
    t_dev = open(os.path.join(path, dsets[1], "dev.json"))

    train_data = json.load(t_train)
    test_data = json.load(t_test)
    dev_data = json.load(t_dev)

    train_relations = list(set([sample["relation"] for sample in train_data]))
    test_relations = list(set([sample["relation"] for sample in test_data]))
    dev_relations = list(set([sample["relation"] for sample in dev_data]))

    print("Check if all test relations are in the train set")
    for rel in test_relations:
        if rel not in train_relations:
            print("Not in: {}".format(rel))
    print()
    print("Ok")

    print("Check if all dev relations are in the train set")
    for rel in dev_relations:
        if rel not in train_relations:
            print("Not in: {}".format(rel))
    print()
    print("Ok")
    print(sorted(train_relations))

    exit()


    print("****** Tacred Data statistics *******")
    for file in files:
        data = json.load(open(os.path.join(path, file), "r"))
        print("{} partition".format(file.split(".")[0]))
        print("No of samples: {}".format(len(data)))
        current_relations = []
        current_entities = []
        for idx, sample in enumerate(data):
            # print(sample["token"])
            # print(sample["subj_start"], sample["subj_end"], sample["subj_type"])
            # print(sample["obj_start"], sample["obj_end"], sample["obj_type"])
            # print(sample["relation"])
            # print(sample["stanford_head"])
            # print()
            # if idx == 10:
            #     break
            current_relations.append(sample["relation"])
            current_entities.extend([sample["subj_type"], sample["obj_type"]])

        relations_counter = collections.Counter(current_relations)
        entities_counter = collections.Counter(current_entities)

        relations_set = set(current_relations)
        entities_set = set(current_entities)
        print("** Relation distribution: {}".format(relations_counter))
        print("Total number of relations: {}".format(sum(relations_counter.values())))
        print("** Entities distribution: {}".format(entities_counter))
        print("** Distinct relations: {}, {}".format(len(relations_set), relations_set))
        print("** Sorted relations: {}, {}".format(len(sorted(relations_set)), sorted(relations_set)))
        print("** Distinct entities: {}, {}".format(len(entities_set), entities_set))
        print("** Sorted entities: {}, {}".format(len(sorted(entities_set)), sorted(entities_set)))
        print()
        print()


# path_test = "/home/frankmtumbuka/Projects/RelCore/data/nyt/test.json"
#     path_train = "/home/frankmtumbuka/Projects/RelCore/data/nyt/train.json"
#     print("Test data")
#     t_data = open(path_test, "r")
#     t_data = json.load(t_data)["data"]
#     relations = []
#     for sample in t_data:
#         relations.append(sample["rel"])
#
#     print("Number of samples: {}".format(len(relations)))
#     distinct_rels = collections.Counter(relations)
#     print("Relation distribution: {}".format(distinct_rels))
#     rel_d = sorted(list(distinct_rels.keys()))
#     print("Distinct relations count: {}".format(len(rel_d)))
#     print(rel_d)
#     print("End of test")
#     print()
#
#     print("Train data")
#     train_data = open(path_train, "r")
#     train_data = json.load(train_data)["data"]
#     train_relations = []
#     for sample in train_data:
#         train_relations.append(sample["rel"])
#
#     print("Number of samples: {}".format(len(train_relations)))
#     train_distinct_rels = collections.Counter(train_relations)
#     print("Relation distribution: {}".format(train_distinct_rels))
#     train_rel_d = sorted(list(train_distinct_rels.keys()))
#     print("Distinct relations count: {}".format(len(train_rel_d)))
#     print(train_rel_d)
#     print("End of train")
#
#     # Check is all test relations are in train set
#     not_in = []
#     for i in rel_d:
#         if i not in train_rel_d:
#             not_in.append(i)
#     if len(not_in) > 0:
#         print("Number of relations in the test set but not train set: {}".format(len(not_in)))
#         print(not_in)
#         for i in not_in:
#             print("{} : {}".format(i, distinct_rels[i]))
#     else:
#         print("All relations are in the train set")
    