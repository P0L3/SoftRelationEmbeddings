# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
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
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.nyt_data_loader as nyt_data_loader
import datapipeline.pretrain_dataloader as pretrain_dataloader
import datapipeline.pretrain_dataset as pretrain_dataset
import datapipeline.pretraining_pairs as pretraining_pairs
import datapipeline.relational_dataset as relational_dataset
import datapipeline.relation_input_example as relation_input_example
import datapipeline.tacred_data_loader as tacred_data_loader
import datapipeline.wikidata_data_loader as wikidata_data_loader
import relcore.classifier.base_classifier as base_classifier
import relcore.classifier.mlp_classifier as mlp_classifier
import relcore.classifier.no_classifier as no_classifier
import relcore.encoder.albert_encoder as albert_encoder
import relcore.encoder.base_encoder as base_encoder
import relcore.encoder.bert_encoder as bert_encoder
import relcore.encoder.roberta_encoder as roberta_encoder
import relcore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import relcore.intermediatelayer.contrastive_loss_layer as contrastive_loss_layer
import relcore.intermediatelayer.no_intermediate_layer as no_intermediate_layer
import relcore.pretraininglosses.contrastive_pretrain_loss as contrastive_pretrain_loss
import relcore.rel_core_model as rel_core_model
import relcore.rel_encoder as rel_encoder
import expbase.util as util
import experiment
import experiment.config as config
import experiment.rel_encoder_checkpoint as rel_encoder_checkpoint
import insanity
import os
import torch
import torch.optim as optim
import torch.utils.data as data
import typing
from transformers import AutoTokenizer


class ComponentFactory(object):
    """Creates components from the user defined configuration for an experiment."""

    @classmethod
    def _create_classifier(cls, conf: config.Config, enc_hidden_size: int) -> base_classifier.BaseClassifier:
        """
        This creates a classifier based on the specified experiment configurations.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            enc_hidden_size (int): The hidden size of the encoder.
        Returns:
            classifier (::class:`base_classifier.BaseClassifier`): The new classifier.
        """
        num_classes = conf.num_classes
        if conf.ace_dataset:
            num_classes = len(experiment.ACE_RELATION_TYPES_MAP)
        if conf.tacred_dataset or conf.tacrev_dataset:
            num_classes = len(experiment.TACRED_RELATION_TYPES_MAP)
        if conf.retacred_dataset:
            num_classes = len(experiment.RETACRED_RELATION_TYPES_MAP)
        if conf.nyt_dataset:
            num_classes = len(experiment.NYT_RELATION_TYPES_MAP)
        if conf.wikidata_dataset:
            num_classes = len(experiment.WIKIDATA_RELATION_TYPES_MAP)

        classifier = no_classifier.NoClassifier(input_size=enc_hidden_size, num_classes=num_classes)
        if conf.mlp_classifier:
            classifier = mlp_classifier.MLPClassifier(
                input_size=enc_hidden_size,
                num_classes=num_classes,
                hidden_layers=conf.classifier_layers,
                dropout_rate=conf.dropout_rate,
                ht=conf.ht,
                rel_mask=conf.rel_mask
            )

        return classifier

    @classmethod
    def _create_intermediate_layer(cls, conf: config.Config) -> base_intermediate_layer.BaseIntermediateLayer:
        """
        This creates an intermediate layer based on the specified experiment configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
        Returns:
            intermediate_layer (::class:`base_intermediate_layer.BaseIntermediateLayer`): The new intermediate layer.
        """

        # The default intermediate layer is no intermediate layer at all.
        intermediate_layer = no_intermediate_layer.NoIntermediateLayer()

        if conf.cp_layer:
            intermediate_layer = contrastive_loss_layer.ContrastiveLossLayer(conf.cp_loss_tau)

        return intermediate_layer

    @classmethod
    def _create_relation_encoder(cls, conf: config.Config, test: bool = False) -> base_encoder.BaseEncoder:
        """
        This creates a relation encoder based on the specified experiment configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            test (bool)
        Returns:
            encoder (::class:`base_encoder.BaseEncoder`): The new encoder.
        """

        # The default entity encoder is the BERT encoder.
        encoder = bert_encoder.BertEncoder(experiment.BERT_BASE_VERSION)

        if conf.albert_enc_rel:
            # Create ALBERT based entity encoder.
            encoder = albert_encoder.AlbertEncoder(experiment.ALBERT_XX_LARGE_VERSION)

        if conf.roberta_enc_rel:
            # Create RoBERTa based entity encoder.
            encoder = roberta_encoder.RoBERTaEncoder(experiment.ROBERTA_LARGE_VERSION)

        if conf.checkpoint is not None and not test:
            print("Loading pretrained encoder from: {}".format(conf.checkpoint))
            checkpoint = rel_encoder_checkpoint.RelEncoderCheckpoint.load(conf.checkpoint)
            encoder.load_state_dict(checkpoint.encoder_state)
            print("Ok")
            print()

        # Freeze encoder if specified in the experiment configuration.
        if conf.freeze_rel_enc:
            for p in encoder.parameters():
                p.requires_grad = False

        return encoder

    @classmethod
    def create_batch(
            cls,
            batch: typing.List[relation_input_example.RelationInputExample],
            mask_token_id: int,
            padding_token_id: int = 0,
            mask_percentage: float = 0.0,
            test: bool = False,
    ) -> typing.Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor
    ]:
        """
        This prepares a batch from a list of input examples.
        Args:
            batch (list[::class:`relation_input_example.RelationInputExample`): A list of input examples.
            mask_token_id (int): The id of the mask token in the vocabulary.
            padding_token_id (int): The id of the padding token in the vocabulary.
            mask_percentage (float): The percentage of tokens to be masked out.
            test (bool): Indicates whether its test time or not.

        Returns:
            input_seq (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            entity_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entities and 0s non-entity tokens, (batch-size x seq-len).
            relation_mask (::class:`torch.LongTensor`): A tensor of relation masks with 1s representing the positions
                of the mask token representing the relations and 0s otherwise, (batch-size x seq-len).
            relation_types (::class:`torch.LongTensor`): A tensor of relation types.
            mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels, (batch-size x seq-len).
        """
        # extract features
        input_seq = []
        head_entity_mask = []
        tail_entity_mask = []
        relation_mask = []
        relation_types = []

        for sample in batch:
            input_seq.append(sample.input_tokens_prompt_idx)
            head_entity_mask.append(sample.head_entity_mask)
            tail_entity_mask.append(sample.tail_entity_mask)
            relation_mask.append(sample.relation_mask)
            relation_types.append([sample.relation_type_idx])

        # Pad sequences
        input_seq = cls.pad_sequences(input_seq, padding_token_id)
        head_entity_mask = cls.pad_sequences(head_entity_mask, 0)
        tail_entity_mask = cls.pad_sequences(tail_entity_mask, 0)
        relation_mask = cls.pad_sequences(relation_mask, 0)

        # Create tensors
        input_seq = torch.LongTensor(input_seq)
        head_entity_mask = torch.LongTensor(head_entity_mask)
        tail_entity_mask = torch.LongTensor(tail_entity_mask)
        relation_mask = torch.LongTensor(relation_mask)
        relation_types = torch.LongTensor(relation_types)

        # Generate MLM labels
        mlm_labels = None
        if not test:
            input_seq, mlm_labels = cls.mask_sequences(
                input_seq,
                mask_token_id,
                head_entity_mask,
                tail_entity_mask,
                relation_mask,
                mask_percentage
            )
        else:
            mlm_labels = input_seq.clone()

        return input_seq, head_entity_mask, tail_entity_mask, relation_mask, relation_types, mlm_labels

    @classmethod
    def create_pretrain_batch(
            cls,
            batch: typing.List[pretraining_pairs.SentencePairs],
            mask_token_id: int,
            padding_token_id: int = 0,
            mask_percentage: float = 0.0,
            test: bool = False,
    ) -> typing.Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor
    ]:
        """
        This prepares a batch from a list of input examples.
        Args:
            batch (list[::class:`pretraining_pairs.SentencePairs`): A list of input examples.
            mask_token_id (int): The id of the mask token in the vocabulary.
            padding_token_id (int): The id of the padding token in the vocabulary.
            mask_percentage (float): The percentage of tokens to be masked out.
            test (bool): Indicates whether its test time or not.

        Returns:
            sent_a_input (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            sent_a_ent_a_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity A and 0s non-entity tokens in the first sentence, (batch-size x seq-len).
            sent_a_ent_b_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity B and 0s non-entity tokens in the first sentence, (batch-size x seq-len).
            sent_a_prompt_mask (::class:`torch.LongTensor`): A tensor of prompt masks with 1s representing the positions
                of mask representing a relation and 0s otherwise in the first sentence, (batch-size x seq-len).
            sent_a_mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels for the first sentence,
                (batch-size x seq-len).
            sent_a_attention_mask (::class:`torch.LongTensor`): A tensor of attention masks with 1s tokens to be
                attended to, the actual tokens in first sentences, and 0s for padding tokens,(batch-size x seq-len).
            sent_b_input (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            sent_b_ent_a_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity A and 0s non-entity tokens in the second sentence, (batch-size x seq-len).
            sent_b_ent_b_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity B and 0s non-entity tokens in the second sentence, (batch-size x seq-len).
            sent_b_prompt_mask (::class:`torch.LongTensor`): A tensor of prompt masks with 1s representing the positions
                of mask representing a relation and 0s otherwise in the second sentence, (batch-size x seq-len).
            sent_b_mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels for the first sentence,
                (batch-size x seq-len).
            sent_b_attention_mask (::class:`torch.LongTensor`): A tensor of attention masks with 1s tokens to be
                attended to, the actual tokens in second sentences, and 0s for padding tokens,(batch-size x seq-len).
            contrastive_labels (::class:`torch.LongTensor`): A tensor of contrastive labels, (batch-size x 1)
        """
        # extract features
        sent_a_input = []
        sent_a_ent_a_mask = []
        sent_a_ent_b_mask = []
        sent_a_prompt_mask = []
        sent_a_mlm_labels = None
        sent_a_attention_mask = []
        sent_b_input = []
        sent_b_ent_a_mask = []
        sent_b_ent_b_mask = []
        sent_b_prompt_mask = []
        sent_b_mlm_labels = None
        sent_b_attention_mask = []
        clusters = []

        for sample in batch:
            sent_a_input.append(sample.first_sent_tokens)
            sent_a_ent_a_mask.append(sample.first_sent_ent_a_mask)
            sent_a_ent_b_mask.append(sample.first_sent_ent_b_mask)
            sent_a_prompt_mask.append(sample.first_sent_prompt_mask)
            sent_b_input.append(sample.second_sent_tokens)
            sent_b_ent_a_mask.append(sample.second_sent_ent_a_mask)
            sent_b_ent_b_mask.append(sample.second_sent_ent_b_mask)
            sent_b_prompt_mask.append(sample.second_sent_prompt_mask)
            clusters.append(sample.cluster_idx)

        # Pad sequences
        sent_a_input = cls.pad_sequences(sent_a_input, padding_token_id)
        sent_a_ent_a_mask = cls.pad_sequences(sent_a_ent_a_mask, 0)
        sent_a_ent_b_mask = cls.pad_sequences(sent_a_ent_b_mask, 0)
        sent_a_prompt_mask = cls.pad_sequences(sent_a_prompt_mask, 0)
        sent_b_input = cls.pad_sequences(sent_b_input, padding_token_id)
        sent_b_ent_a_mask = cls.pad_sequences(sent_b_ent_a_mask, 0)
        sent_b_ent_b_mask = cls.pad_sequences(sent_b_ent_b_mask, 0)
        sent_b_prompt_mask = cls.pad_sequences(sent_b_prompt_mask, 0)

        # Attention masks
        sent_a_attention_mask = [[int(token != padding_token_id) for token in seq] for seq in sent_a_input]
        sent_b_attention_mask = [[int(token != padding_token_id) for token in seq] for seq in sent_b_input]

        # Create tensors
        sent_a_input = torch.LongTensor(sent_a_input)
        sent_a_ent_a_mask = torch.LongTensor(sent_a_ent_a_mask)
        sent_a_ent_b_mask = torch.LongTensor(sent_a_ent_b_mask)
        sent_a_prompt_mask = torch.LongTensor(sent_a_prompt_mask)
        sent_a_attention_mask = torch.LongTensor(sent_a_attention_mask)
        sent_b_input = torch.LongTensor(sent_b_input)
        sent_b_ent_a_mask = torch.LongTensor(sent_b_ent_a_mask)
        sent_b_ent_b_mask = torch.LongTensor(sent_b_ent_b_mask)
        sent_b_prompt_mask = torch.LongTensor(sent_b_prompt_mask)
        sent_b_attention_mask = torch.LongTensor(sent_b_attention_mask)

        # Generate contrastive loss labels
        contrastive_labels = cls.create_contrastive_labels(clusters)
        contrastive_labels = torch.LongTensor(contrastive_labels)

        # # Generate MLM labels
        # if not test:
        #     sent_a_input, sent_a_mlm_labels = cls.mask_sequences(
        #         sent_a_input,
        #         mask_token_id,
        #         sent_a_ent_a_mask,
        #         sent_a_ent_b_mask,
        #         sent_a_prompt_mask,
        #         mask_percentage
        #     )
        #
        #     sent_b_input, sent_b_mlm_labels = cls.mask_sequences(
        #         sent_b_input,
        #         mask_token_id,
        #         sent_b_ent_a_mask,
        #         sent_b_ent_b_mask,
        #         sent_b_prompt_mask,
        #         mask_percentage
        #     )
        # else:
        #     sent_a_mlm_labels = sent_a_input.clone()
        #     sent_b_mlm_labels = sent_b_input.clone()

        return (
            sent_a_input,
            sent_a_ent_a_mask,
            sent_a_ent_b_mask,
            sent_a_prompt_mask,
            sent_a_mlm_labels,
            sent_a_attention_mask,
            sent_b_input,
            sent_b_ent_a_mask,
            sent_b_ent_b_mask,
            sent_b_prompt_mask,
            sent_b_mlm_labels,
            sent_b_attention_mask,
            contrastive_labels
        )

    @classmethod
    def create_nyt_pretrain_batch(
            cls,
            batch: typing.List[pretraining_pairs.SentencePairs],
            mask_token_id: int,
            padding_token_id: int = 0,
            mask_percentage: float = 0.0,
            test: bool = False,
    ) -> typing.Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor
    ]:
        """
        This prepares a batch from a list of input examples.
        Args:
            batch (list[::class:`pretraining_pairs.SentencePairs`): A list of input examples.
            mask_token_id (int): The id of the mask token in the vocabulary.
            padding_token_id (int): The id of the padding token in the vocabulary.
            mask_percentage (float): The percentage of tokens to be masked out.
            test (bool): Indicates whether its test time or not.

        Returns:
            sent_input (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            sent_ent_a_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity A and 0s non-entity tokens in the sentence, (batch-size x seq-len).
            sent_ent_b_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity B and 0s non-entity tokens in the sentence, (batch-size x seq-len).
            sent_prompt_mask (::class:`torch.LongTensor`): A tensor of prompt masks with 1s representing the positions
                of mask representing a relation and 0s otherwise in the sentence, (batch-size x seq-len).
            sent_mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels for the sentence,
                (batch-size x seq-len).
            sent_attention_mask (::class:`torch.LongTensor`): A tensor of attention masks with 1s tokens to be
                attended to, the actual tokens in sentences, and 0s for padding tokens,(batch-size x seq-len).
            contrastive_labels (::class:`torch.LongTensor`): A tensor of contrastive labels, (batch-size x 1)
        """
        # extract features
        sent_input = []
        sent_ent_a_mask = []
        sent_ent_b_mask = []
        sent_prompt_mask = []
        sent_mlm_labels = None
        sent_attention_mask = []
        relation_types = []

        for sample in batch:
            sent_input.append(sample.input_tokens_prompt_idx)
            sent_ent_a_mask.append(sample.head_entity_mask)
            sent_ent_b_mask.append(sample.tail_entity_mask)
            sent_prompt_mask.append(sample.relation_mask)
            relation_types.append(sample.relation_type_idx)

        # Pad sequences
        sent_input = cls.pad_sequences(sent_input, padding_token_id)
        sent_ent_a_mask = cls.pad_sequences(sent_ent_a_mask, 0)
        sent_ent_b_mask = cls.pad_sequences(sent_ent_b_mask, 0)
        sent_prompt_mask = cls.pad_sequences(sent_prompt_mask, 0)

        # Attention masks
        sent_attention_mask = [[int(token != padding_token_id) for token in seq] for seq in sent_input]

        # Create tensors
        sent_input = torch.LongTensor(sent_input)
        sent_ent_a_mask = torch.LongTensor(sent_ent_a_mask)
        sent_ent_b_mask = torch.LongTensor(sent_ent_b_mask)
        sent_prompt_mask = torch.LongTensor(sent_prompt_mask)
        sent_attention_mask = torch.LongTensor(sent_attention_mask)

        # Generate contrastive loss labels
        contrastive_labels = cls.create_contrastive_labels(relation_types)
        contrastive_labels = torch.LongTensor(contrastive_labels)

        # # Generate MLM labels
        # if not test:
        #     sent_a_input, sent_a_mlm_labels = cls.mask_sequences(
        #         sent_a_input,
        #         mask_token_id,
        #         sent_a_ent_a_mask,
        #         sent_a_ent_b_mask,
        #         sent_a_prompt_mask,
        #         mask_percentage
        #     )
        #
        #     sent_b_input, sent_b_mlm_labels = cls.mask_sequences(
        #         sent_b_input,
        #         mask_token_id,
        #         sent_b_ent_a_mask,
        #         sent_b_ent_b_mask,
        #         sent_b_prompt_mask,
        #         mask_percentage
        #     )
        # else:
        #     sent_a_mlm_labels = sent_a_input.clone()
        #     sent_b_mlm_labels = sent_b_input.clone()

        return (
            sent_input,
            sent_ent_a_mask,
            sent_ent_b_mask,
            sent_prompt_mask,
            sent_mlm_labels,
            sent_attention_mask,
            contrastive_labels
        )

    @classmethod
    def create_dataset(cls, conf: config.Config, dev: bool = False, test: bool = False) -> data.Dataset:
        """
        This creates a dataset specified by the user in the experiment configuration.
        Args:
            conf (::class:`config.Config`): The user defined configuration for the experiment.
            dev (bool): Specifies whether to load the dev partition of the specified dataset.
            test (bool): Specifies whether to load the test partition of the specified dataset.
        Returns:
            dataset (::class:`torch.Dataset`): The created dataset.
        """
        dataset = None
        pickle_file = None
        # If pre-train
        if conf.pretrain:
            if conf.nyt_dataset:
                data_path = os.path.join(conf.data_dir, experiment.DATASET_DIRS["nyt"])
                print("Loading data from {}...".format(data_path))
                # Default tokenizer is for the BERT model
                tokenizer_name = experiment.BERT_BASE_VERSION
                if conf.albert_enc_rel:
                    tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
                if conf.roberta_enc_rel:
                    tokenizer_name = experiment.ROBERTA_LARGE_VERSION
                pickle_file = f"{data_path}train.prompt.{tokenizer_name}.data.pickle"

                if os.path.isfile(pickle_file):
                    print("Loading data from {}.".format(pickle_file))
                    dataset = util.better_pickle.pickle_load(pickle_file)
                else:
                    # Default tokenizer is for the BERT model
                    print("Not pickle")
                    tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                    if conf.albert_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                    if conf.roberta_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                    print("Loading data from {}...".format(data_path))
                    data_loader = nyt_data_loader.NYTDataLoader(data_path, dev=dev, test=test)
                    rel_types = experiment.NYT_RELATION_TYPES_MAP
                    dataset = relational_dataset.RelationalDataset(
                        data_loader,
                        rel_types,
                        tokenizer,
                        relation_prompt=conf.relation_prompt
                    )
                    if conf.pickle_data or test or dev:
                        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                        util.better_pickle.pickle_dump(dataset, pickle_file)
                        print("OK")
            elif conf.wikidata_dataset:
                data_path = os.path.join(conf.data_dir, experiment.DATASET_DIRS["wikidata"])
                print("Loading data from {}...".format(data_path))
                # Default tokenizer is for the BERT model
                tokenizer_name = experiment.BERT_BASE_VERSION
                if conf.albert_enc_rel:
                    tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
                if conf.roberta_enc_rel:
                    tokenizer_name = experiment.ROBERTA_LARGE_VERSION
                pickle_file = f"{data_path}train.prompt.{tokenizer_name}.data.pickle"

                if os.path.isfile(pickle_file):
                    print("Loading data from {}.".format(pickle_file))
                    dataset = util.better_pickle.pickle_load(pickle_file)
                else:
                    # Default tokenizer is for the BERT model
                    print("Not pickle")
                    tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                    if conf.albert_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                    if conf.roberta_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                    print("Loading data from {}...".format(data_path))
                    data_loader = wikidata_data_loader.WikidataDataLoader(data_path, dev=dev, test=test)
                    rel_types = experiment.WIKIDATA_RELATION_TYPES_MAP
                    dataset = relational_dataset.RelationalDataset(
                        data_loader,
                        rel_types,
                        tokenizer,
                        relation_prompt=conf.relation_prompt
                    )
                    if conf.pickle_data or test or dev:
                        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                        util.better_pickle.pickle_dump(dataset, pickle_file)
                        print("OK")
            elif conf.tacred_dataset or conf.tacrev_dataset or conf.retacred_dataset:
                data_dir = None
                if conf.tacred_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["tacred"])
                if conf.retacred_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["retacred"])
                if conf.tacrev_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["tacrev"])
                data_path = data_dir

                if dev:
                    data_path = os.path.join(data_path, "dev")
                elif test:
                    data_path = os.path.join(data_path, "test")
                else:
                    data_path = os.path.join(data_path, "train")

                if conf.relation_prompt:
                    data_path = f"{data_path}.prompt"

                # Default tokenizer is for the BERT model
                tokenizer_name = experiment.BERT_BASE_VERSION
                if conf.albert_enc_rel:
                    tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
                if conf.roberta_enc_rel:
                    tokenizer_name = experiment.ROBERTA_LARGE_VERSION

                pickle_file = f"{data_path}.{tokenizer_name}.data.pickle"
                if os.path.isfile(pickle_file):
                    print("Loading pre-training data from {}.".format(pickle_file))
                    dataset = util.better_pickle.pickle_load(pickle_file)
                else:
                    # Default tokenizer is for the BERT model
                    tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                    if conf.albert_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                    if conf.roberta_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                    print("Loading pre-training data from {}...".format(data_path))
                    data_loader = tacred_data_loader.TACREDDataLoader(data_dir, dev=dev, test=test)

                    rel_types = None
                    if conf.retacred_dataset:
                        rel_types = experiment.RETACRED_RELATION_TYPES_MAP
                    else:
                        rel_types = experiment.TACRED_RELATION_TYPES_MAP

                    dataset = relational_dataset.RelationalDataset(
                        data_loader,
                        rel_types,
                        tokenizer,
                        relation_prompt=conf.relation_prompt
                    )
                    if conf.pickle_data or test or dev:
                        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                        util.better_pickle.pickle_dump(dataset, pickle_file)
                        print("OK")
            else:
                data_path = os.path.join(conf.data_dir, experiment.DATASET_DIRS["pretrain"])
                print("Loading data from {}...".format(data_path))

                # Default tokenizer is for the BERT model
                tokenizer_name = experiment.BERT_BASE_VERSION
                if conf.albert_enc_rel:
                    tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
                if conf.roberta_enc_rel:
                    tokenizer_name = experiment.ROBERTA_LARGE_VERSION

                pickle_file = f"{data_path}{tokenizer_name}.data.pickle"
                if os.path.isfile(pickle_file):
                    print("Loading data from {}.".format(pickle_file))
                    dataset = util.better_pickle.pickle_load(pickle_file)
                else:
                    # Default tokenizer is for the BERT model
                    tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                    if conf.albert_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                    if conf.roberta_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                    print("Loading data from {}...".format(data_path))
                    data_loader = pretrain_dataloader.PretrainDataLoader(data_path)
                    dataset = pretrain_dataset.PretrainDataset(data_loader, tokenizer)
                    print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                    util.better_pickle.pickle_dump(dataset, pickle_file)
                    print("OK")
        else:
            if (
                    conf.tacred_dataset or
                    conf.retacred_dataset or
                    conf.tacrev_dataset or
                    conf.nyt_dataset or
                    conf.wikidata_dataset
            ):
                data_dir = None
                if conf.tacred_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["tacred"])
                if conf.retacred_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["retacred"])
                if conf.tacrev_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["tacrev"])
                if conf.nyt_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["nyt"])

                if conf.wikidata_dataset:
                    data_dir = os.path.join(conf.data_dir, experiment.DATASET_DIRS["wikidata"])

                data_path = data_dir

                if dev:
                    data_path = os.path.join(data_path, "dev")
                elif test:
                    data_path = os.path.join(data_path, "test")
                else:
                    data_path = os.path.join(data_path, "train")

                if conf.relation_prompt:
                    data_path = f"{data_path}.prompt"

                # Default tokenizer is for the BERT model
                tokenizer_name = experiment.BERT_BASE_VERSION
                if conf.albert_enc_rel:
                    tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
                if conf.roberta_enc_rel:
                    tokenizer_name = experiment.ROBERTA_LARGE_VERSION

                pickle_file = f"{data_path}.{tokenizer_name}.data.pickle"
                if os.path.isfile(pickle_file):
                    print("Loading data from {}.".format(pickle_file))
                    dataset = util.better_pickle.pickle_load(pickle_file)
                else:
                    # Default tokenizer is for the BERT model
                    tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                    if conf.albert_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                    if conf.roberta_enc_rel:
                        tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                    print("Loading data from {}...".format(data_path))
                    data_loader = None
                    if conf.nyt_dataset:
                        data_loader = nyt_data_loader.NYTDataLoader(data_dir, dev=dev, test=test)
                    elif conf.wikidata_dataset:
                        data_loader = wikidata_data_loader.WikidataDataLoader(data_dir, dev=dev, test=test)
                    else:
                        data_loader = tacred_data_loader.TACREDDataLoader(data_dir, dev=dev, test=test)

                    rel_types = None
                    if conf.retacred_dataset:
                        rel_types = experiment.RETACRED_RELATION_TYPES_MAP
                    elif conf.nyt_dataset:
                        rel_types = experiment.NYT_RELATION_TYPES_MAP
                    elif conf.wikidata_dataset:
                        rel_types = experiment.WIKIDATA_RELATION_TYPES_MAP
                    else:
                        rel_types = experiment.TACRED_RELATION_TYPES_MAP

                    dataset = relational_dataset.RelationalDataset(
                        data_loader,
                        rel_types,
                        tokenizer,
                        relation_prompt=conf.relation_prompt
                    )
                    if conf.pickle_data or test or dev:
                        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                        util.better_pickle.pickle_dump(dataset, pickle_file)
                        print("OK")
            else:
                print("To be implemented for other datasets")
                print("Ok")
        return dataset

    @classmethod
    def create_model(
            cls,
            conf: config.Config,
            test: bool = False
    ) -> typing.Union[rel_core_model.RelCoreModel, rel_encoder.RelEncoder]:
        """
        This creates the RelCoreModel that encapsulates all the modules specified in the experiment
        configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            test (bool):
        Returns:
            RelCoreModel (::class:`rel_core_model.RelCoreModel`): The new RelCoreModel.
        """
        model = None
        if conf.pretrain:
            encoder = cls._create_relation_encoder(conf)
            loss_layer = contrastive_pretrain_loss.ContrastivePretrainLoss(
                encoder.hidden_size(),
                h_and_t=conf.ht,
                rel_mask=conf.rel_mask,
                temperature=conf.cp_loss_tau
            )
            model = rel_encoder.RelEncoder(encoder, loss_layer)

        else:
            intermediate_layer = cls._create_intermediate_layer(conf)  # Creates an intermediate layer
            encoder = cls._create_relation_encoder(conf, test)  # Creates a relation encoder
            # classifier_hidden_size = 0
            # if conf.ht and conf.rel_mask:
            #     classifier_hidden_size = encoder.hidden_size() * 3
            # elif conf.ht:
            #     classifier_hidden_size = encoder.hidden_size() + encoder.hidden_size()
            # elif conf.rel_mask:
            #     classifier_hidden_size = encoder.hidden_size()
            # else:
            #     classifier_hidden_size = encoder.hidden_size()
            classifier = cls._create_classifier(conf, encoder.hidden_size())  # Creates a classifier.

            model = rel_core_model.RelCoreModel(classifier, encoder, intermediate_layer, conf.ht, conf.rel_mask)

        return model

    @classmethod
    def create_optimizer(cls, conf: config.Config, model: rel_core_model.RelCoreModel) -> optim.Optimizer:
        """
        This creates an optimizer for training the model.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            model (::class:`encore_model.EnCoreModel`): The model to be trained.

        Returns:
            optimizer (::class:`optim.Optimizer`): The optimizer for training the model.
        """

        return optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=conf.learning_rate
        )

    @classmethod
    def mask_sequences(
            cls,
            input_seq: torch.LongTensor,
            mask_token_id: int,
            head_entity_mask: torch.LongTensor = None,
            tail_entity_mask: torch.LongTensor = None,
            relation_mask: torch.LongTensor = None,
            mask_percentage: float = 0.00
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        This masks a specified percentage of tokens in the input sequence.
        Args:
            input_seq (::class:`torch.LongTensor`): The input ids.
            mask_token_id (int): The id of the mask token in the vocabulary.
            head_entity_mask (::class:`torch.LongTensor`): The mask indicating the positions of entities.
            tail_entity_mask:
            relation_mask (::class:`torch.LongTensor`): The mask indicating the position of the relation mask
            mask_percentage (float): The percentage of tokens to be masked out.
        Returns:

        """
        # Sanitize args.
        insanity.sanitize_type("input_seq", input_seq, torch.Tensor)
        insanity.sanitize_type("mask_token_id", mask_token_id, int)
        insanity.sanitize_range("mask_token_id", mask_token_id, minimum=0)
        insanity.sanitize_type("mask_percentage", mask_percentage, float)
        insanity.sanitize_range("mask_percentage", mask_percentage, minimum=0.00)
        if head_entity_mask is not None:
            insanity.sanitize_type("head_entity_mask", head_entity_mask, torch.Tensor)
            assert input_seq.shape == head_entity_mask.shape

        # Create labels, a clone of input_seq
        labels = input_seq.clone()

        # Sample tokens for masking.
        probability_matrix = torch.full(labels.shape, mask_percentage)

        masked_indices = None
        if head_entity_mask is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else:
            masked_indices = torch.bernoulli(probability_matrix).bool() & \
                             (~(head_entity_mask.bool())) & \
                             (~(relation_mask.bool())) & \
                             (~(tail_entity_mask.bool()))

        labels[~masked_indices] = -100

        input_seq[masked_indices] = mask_token_id

        return input_seq, labels

    @classmethod
    def pad_sequences(cls, input_seq: list, padding_token: typing.Union[str, int]) -> list:
        """

        Args:
            input_seq:
            padding_token:

        Returns:

        """
        max_len = max(len(s) for s in input_seq)
        return [s + [padding_token] * (max_len - len(s)) for s in input_seq]

    @classmethod
    def create_contrastive_labels(cls, raw_labels: typing.List) -> typing.List[int]:
        """
        This creates contrastive labels by assigning values ranging from 0
        Args:
            raw_labels (list[int]): The raw labels from the dataset.

        Returns:
            contrastive_labels (list[int]): The generated contrastive labels.
        """
        labels_set = set(raw_labels)
        label_mapping = {}
        for idx, i in enumerate(labels_set):
            label_mapping[i] = idx
        return [label_mapping[i] for i in raw_labels]

     @classmethod
    def create_my_contrastive_collate_fn(cls, tokenizer: AutoTokenizer):
        def collate_fn(batch):
            # batch is a list of (anchor_text, [candidate_texts], [levels])
            
            # Replace placeholder [MASK] with the real mask token
            mask_token = tokenizer.mask_token
            batch_tuples = [
                (anchor.replace("[MASK]", mask_token), 
                 [cand.replace("[MASK]", mask_token) for cand in cands], 
                 levels)
                for anchor, cands, levels in batch
            ]
            
            batch_anchors = [t[0] for t in batch_tuples]
            batch_candidates_nested = [t[1] for t in batch_tuples]
            batch_levels_nested = [t[2] for t in batch_tuples]
            
            # This stores how many candidates each anchor has
            num_candidates_per_anchor = [len(cands) for cands in batch_candidates_nested]
            
            # Flatten lists for single-pass tokenization
            batch_candidates_flat = [cand for sublist in batch_candidates_nested for cand in sublist]
            batch_levels_flat = [level for sublist in batch_levels_nested for level in sublist]
            
            if not batch_anchors: return None

            anchor_inputs = tokenizer(batch_anchors, padding=True, truncation=True, return_tensors="pt")
            candidate_inputs = tokenizer(batch_candidates_flat, padding=True, truncation=True, return_tensors="pt")
            levels_tensor = torch.tensor(batch_levels_flat, dtype=torch.long)
            
            return anchor_inputs, candidate_inputs, levels_tensor, num_candidates_per_anchor
            
        return collate_fn