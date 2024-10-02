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
__date__ = "08 Aug 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.pretraining_pairs as pretraining_pairs
import json
import os
import typing


class PretrainDataLoader(base_data_loader.BaseDataLoader):
    """This loads the pretraining data from the filesystem."""

    DATA_FILE = "samples.json"
    """str: The name of the file with training samples."""

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: See `base_data_loader.BaseDataLoader.__init__.
            **kwargs: See `base_data_loader.BaseDataLoader.__init__.
        """
        super().__init__(*args, **kwargs)

    def load(self) -> typing.List[pretraining_pairs.SentencePairs]:
        """See meth::`base_data_loader.BaseDataLoader.load()`"""
        samples = []

        # Load data
        data = open(os.path.join(self._data_path, self.DATA_FILE), "r")
        data = json.load(data)["all"]

        for pair_idx, pair in enumerate(data):
            if len(pair["sentence_a"]["clusters"]) != len(pair["sentence_b"]["clusters"]):
                # If the set of clusters in sentence A has size 2, base on sentence A, else, base on sentence 2.
                if len(set(pair["sentence_a"]["clusters"])) == 2:
                    new_pairs = []
                    new_pair = pretraining_pairs.SentencePairs(pair["sentence_a"]["text"], pair["sentence_b"]["text"])
                    new_pair.cluster_idx = pair_idx
                    for idx, cluster in enumerate(pair["sentence_a"]["clusters"]):
                        # Check if the custer is repeating either in a or b
                        count_a = pair["sentence_a"]["clusters"].count(cluster)
                        count_b = pair["sentence_b"]["clusters"].count(cluster)

                        if idx == 0 and count_b == 1:
                            # Set values for sent_a_ent_a and sent_b_ent_a
                            new_pair.first_sent_ent_a = [
                                pair["sentence_a"]["start_pos"][idx],
                                pair["sentence_a"]["end_pos"][idx]
                            ]
                            similar_index = pair["sentence_b"]["clusters"].index(cluster)
                            new_pair.second_sent_ent_a = [
                                pair["sentence_b"]["start_pos"][similar_index],
                                pair["sentence_b"]["end_pos"][similar_index]
                            ]

                        # TODO! Repeat on entity A
                        if idx == 1 and count_b == 1:
                            # Set values for sent_a_ent_b and sent_b_ent_b
                            new_pair.first_sent_ent_b = [
                                pair["sentence_a"]["start_pos"][idx],
                                pair["sentence_a"]["end_pos"][idx]
                            ]
                            similar_index = pair["sentence_b"]["clusters"].index(cluster)
                            new_pair.second_sent_ent_b = [
                                pair["sentence_b"]["start_pos"][similar_index],
                                pair["sentence_b"]["end_pos"][similar_index]
                            ]
                            try:
                                a = new_pair.first_sent[new_pair.first_sent_ent_a[0]:new_pair.first_sent_ent_a[1]]
                                b = new_pair.first_sent[new_pair.first_sent_ent_b[0]:new_pair.first_sent_ent_b[1]]
                                aa = new_pair.second_sent[new_pair.second_sent_ent_a[0]:new_pair.second_sent_ent_a[1]]
                                bb = new_pair.second_sent[new_pair.second_sent_ent_b[0]:new_pair.second_sent_ent_b[1]]
                                new_pairs.append(new_pair)
                            except Exception as e:
                                continue

                        # Repeat on entity B
                        if idx == 1 and count_b > 1:
                            # Set values for sent_a_ent_b and sent_b_ent_b but repeat values sent_a_ent_a and sent_b_ent_a
                            new_pair.first_sent_ent_b = [pair["sentence_a"]["start_pos"][idx],
                                                     pair["sentence_a"]["end_pos"][idx]]
                            count = 0
                            for b_idx, b_cluster in enumerate(pair["sentence_b"]["clusters"]):
                                if b_cluster == cluster:
                                    p = pretraining_pairs.SentencePairs(new_pair.first_sent, new_pair.second_sent)
                                    p.cluster_idx = new_pair.cluster_idx
                                    p.first_sent_ent_a = new_pair.first_sent_ent_a
                                    p.second_sent_ent_a = new_pair.second_sent_ent_a
                                    p.first_sent_ent_b = new_pair.first_sent_ent_b
                                    p.second_sent_ent_b = [
                                        pair["sentence_b"]["start_pos"][b_idx],
                                        pair["sentence_b"]["end_pos"][b_idx]
                                    ]
                                    try:
                                        a = p.first_sent[p.first_sent_ent_a[0]:p.first_sent_ent_a[1]]
                                        b = p.first_sent[p.first_sent_ent_b[0]:p.first_sent_ent_b[1]]
                                        aa = p.second_sent[p.second_sent_ent_a[0]:p.second_sent_ent_a[1]]
                                        bb = p.second_sent[p.second_sent_ent_b[0]:p.second_sent_ent_b[1]]
                                        new_pairs.append(p)
                                    except Exception as e:
                                        continue

                    samples.extend(new_pairs)
                else:
                    # TODO!
                    pass
            else:
                new_pair = pretraining_pairs.SentencePairs(pair["sentence_a"]["text"], pair["sentence_b"]["text"])
                new_pair.cluster_idx = pair_idx
                for idx, cluster in enumerate(pair["sentence_a"]["clusters"]):
                    if idx == 0:
                        new_pair.first_sent_ent_a = [pair["sentence_a"]["start_pos"][idx],
                                                 pair["sentence_a"]["end_pos"][idx]]
                        # If the count is one, then it has one index
                        similar_index = pair["sentence_b"]["clusters"].index(cluster)
                        new_pair.second_sent_ent_a = [
                            pair["sentence_b"]["start_pos"][similar_index],
                            pair["sentence_b"]["end_pos"][similar_index]
                        ]
                    if idx == 1:
                        new_pair.first_sent_ent_b = [pair["sentence_a"]["start_pos"][idx],
                                                 pair["sentence_a"]["end_pos"][idx]]
                        # If the count is one, then it has one index
                        similar_index = pair["sentence_b"]["clusters"].index(cluster)
                        new_pair.second_sent_ent_b = [
                            pair["sentence_b"]["start_pos"][similar_index],
                            pair["sentence_b"]["end_pos"][similar_index]
                        ]
                try:
                    a = new_pair.first_sent[new_pair.first_sent_ent_a[0]:new_pair.first_sent_ent_a[1]]
                    b = new_pair.first_sent[new_pair.first_sent_ent_b[0]:new_pair.first_sent_ent_b[1]]
                    aa = new_pair.second_sent[new_pair.second_sent_ent_a[0]:new_pair.second_sent_ent_a[1]]
                    bb = new_pair.second_sent[new_pair.second_sent_ent_b[0]:new_pair.second_sent_ent_b[1]]
                    samples.append(new_pair)
                except Exception as e:
                    continue

        return samples

