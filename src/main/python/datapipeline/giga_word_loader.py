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
__date__ = "02 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.cluster as cluster
import datapipeline.relation_input_example as relation_input_example
import datapipeline.sentence as sentence
import datapipeline.story as story
import json
import typing
import os
from tqdm import tqdm


class Sentence(object):
    def __init__(self, text: str):
        self.text = text
        self.clusters = []
        self.start_pos = []
        self.end_pos = []


class GigaWordLoader(base_data_loader.BaseDataLoader):
    """This loads the Giga word corpus."""
    output_file = "samples.json"

    def load(self) -> typing.Union[typing.List[story.Story], typing.List[relation_input_example.RelationInputExample]]:
        stories = []  # To hold all stories.

        # Find files in the data dir
        file_names = []
        for root, dirs, files in os.walk(self._data_path):
            for file in files:
                if file.endswith(".json"):
                    file_names.append(os.path.join(root, file))

        # Loop through files
        for file_idx, filename in enumerate(file_names):
            data = open(filename)
            print("Loading data from: {}".format(filename))
            data = json.load(data)
            data = data["stories"]
            sent_pairs = []
            for story_idx, current_story in enumerate(data):
                current_story = [line.rstrip() for line in current_story]
                current_story = " ".join(current_story)
                current_story = current_story.replace("``", '"')
                current_story = current_story.replace("''", '"')
                try:
                    doc = self._nlp(current_story)
                except Exception as e:
                    continue
                try:
                    sentences = []
                    for c_idx, cluster in enumerate(doc._.coref_clusters):
                        for s, e in cluster:
                            for sent in doc.sents:
                                if sent.start_char <= s and e <= sent.end_char:
                                    start_pos = s - sent.start_char
                                    end_pos = e - sent.start_char
                                    new_sent = Sentence(sent.text)
                                    new_sent.start_pos.append(start_pos)
                                    new_sent.end_pos.append(end_pos)
                                    new_sent.clusters.append(c_idx)
                                    sentences.append(new_sent)
                    # print("Redundant sentences: {}".format(len(sentences)))
                    # Remove redundancies
                    non_redundant = []
                    for sent in sentences:
                        # Check if the sentence is already considered
                        is_checked = [int(sent.text == r_s.text) for r_s in non_redundant]
                        if len(non_redundant) == 0 or sum(is_checked) == 0:
                            non_redundant.append(sent)
                        else:
                            index = is_checked.index(1)
                            non_redundant[index].end_pos.extend(sent.end_pos)
                            non_redundant[index].start_pos.extend(sent.start_pos)
                            non_redundant[index].clusters.extend(sent.clusters)

                    # print("Non-redundant sentences: {}".format(len(non_redundant)))

                    # Sentences that belong to multiple clusters
                    needed_sents = []
                    for sent in non_redundant:
                        if len(set(sent.clusters)) > 1:
                            needed_sents.append(sent)

                    # print("Needed sentences: {}".format(len(needed_sents)))

                    most_need_sents = []
                    for s in needed_sents:
                        c = []
                        for i in needed_sents:
                            if set(s.clusters).issubset(set(i.clusters)) and (i.clusters != s.clusters):
                                # Remove redundant pairs
                                is_in = [int(set(p1.clusters + p2.clusters) == set(s.clusters + i.clusters)) for p1, p2 in
                                         most_need_sents]
                                if sum(is_in) == 0:
                                    c.append((s, i))
                        most_need_sents.extend(c)

                    # print("Most needed sets: {}".format(len(most_need_sents)))
                    for pair in most_need_sents:
                        a, b = pair
                        sent_pairs.append(
                            {
                                "sentence_a": {
                                    "text": a.text,
                                    "clusters": a.clusters,
                                    "start_pos": a.start_pos,
                                    "end_pos": a.end_pos
                                },
                                "sentence_b": {
                                    "text": b.text,
                                    "clusters": b.clusters,
                                    "start_pos": b.start_pos,
                                    "end_pos": b.end_pos
                                }
                            }
                        )
                except Exception as e:
                    continue

                # update the protocol file
                if (story_idx % 500) == 0 or story_idx == (len(data) - 1):
                    saved_file_path = os.path.join(self._data_path, self.output_file)
                    if os.path.isfile(saved_file_path):
                        with open(saved_file_path, "r") as f:
                            proto = json.load(f)
                    else:
                        proto = {"all": []}

                    proto["all"].extend(sent_pairs)
                    with open(saved_file_path, "w") as f:
                        json.dump(proto, f, indent=4)
                    sent_pairs = []
                if story_idx == 500:
                    print("Processed {}/{} in file {}/{}".format(story_idx, len(data), file_idx, len(file_names)))
            # if file_idx == 5:
            #     exit()

