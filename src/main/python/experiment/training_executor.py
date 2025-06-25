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

import collections.abc as collections
import expbase as xb
import expbase.util as util
import torch
import experiment.component_factory as component_factory
import experiment.encore_model_checkpoint as encore_model_checkpoint
import experiment.rel_encoder_checkpoint as rel_encoder_checkpoint
import functools
import numpy as np
import operator
import os
import torch.nn as nn
import torch.nn.utils as utils
import torch.utils.data as data


class TrainingExecutor(xb.TrainingExecutor):
    """This implements the training routing for an experiment based on the configurations given."""

    PRETRAIN_PROTO = "pretrain.proto"

    def __init__(self, *args, **kwargs):
        """This creates an instance of the `TrainingExecutor`."""
        super().__init__(*args, **kwargs)

        # Attributes
        self._cross_entropy_loss = None
        self._dataset = None
        self._model = None
        self._optimizer = None
        self._start_epoch = 0

    # Methods

    def _init(self) -> None:
        # Call routines in the Component factory to create the attributes above
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._dataset = component_factory.ComponentFactory.create_dataset(self._conf)
        self._model = component_factory.ComponentFactory.create_model(self._conf)
        self._optimizer = component_factory.ComponentFactory.create_optimizer(self._conf, self._model)
        # create a helper for creating and maintaining checkpoints
        self._ckpt_saver = xb.CheckpointSaver(
            target_dir=self._conf.results_dir,
            filename_pattern="after-{steps}.ckpt"
        )

    def _run_training(self) -> None:
        
        # --- START OF MODIFICATIONS ---
        collate_function = None
        if self._conf.my_custom_dataset:
            print("Using custom dataset and collate function.")
            # We need the tokenizer to create the collate function
            # Determine which tokenizer to use based on the model config
            if self._conf.albert_enc_rel:
                tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
            elif self._conf.roberta_enc_rel:
                tokenizer_name = experiment.ROBERTA_LARGE_VERSION
            else:
                tokenizer_name = experiment.BERT_BASE_VERSION
                
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # VERY IMPORTANT: The repo's RelationEncoder adds special tokens and resizes the model's
            # embedding layer. We MUST ensure our tokenizer and model are synchronized.
            special_tokens_to_add = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
            self._model.encoder.encoder.resize_token_embeddings(len(tokenizer))
            print("Tokenizer and model embedding layer resized for special tokens.")

            # Create our custom collate function
            collate_function = component_factory.ComponentFactory.create_my_contrastive_collate_fn(tokenizer)
        else:
            # Original logic for all other datasets
            collate_function = lambda x: functools.reduce(operator.add, x)

        data_loader = data.DataLoader(
            self._dataset,
            batch_size=self._conf.batch_size,
            collate_fn=collate_function,
            shuffle=True,
            drop_last=True # Keep this to ensure batches are uniform
        )
        # --- END OF MODIFICATIONS ---

        print("Num of trainable params: {}".format(sum(p.numel() for p in self._model.parameters() if p.requires_grad)))
        print("Size of dataset: {}".format(len(self._dataset)))
        print("Num of iterations per epoch: {}".format(len(data_loader)))

        # Move model to gpu if training on a gpu is specified
        if self._conf.gpu:
            self._model.cuda()
        print("Starting model training...")
        epoch_durations = []
        num_steps = 0
        losses = []
        for epoch in range(self._conf.num_epochs):
            with util.Timer("finished epoch", terminal_break=True) as epoch_timer:
                # Epoch start.....
                # Print epoch header.
                util.printing_tools.print_header("Epoch {}".format(epoch), level=0)

                iteration_durations = []
                for iteration_idx, batch in enumerate(data_loader):
                    # Check whether to print details or not.
                    print_details = (iteration_idx + 1) % self._conf.print_int == 0
                    with util.Timer(
                            "finished iteration",
                            skip_output=not print_details,
                            terminal_break=True
                    ) as iteration_timer:
                        if print_details:
                            util.printing_tools.print_header("Iteration {}".format(iteration_idx), level=1)
                        if self._conf.pretrain:
                            
                            # --- NEW LOGIC BRANCH FOR YOUR DATASET ---
                            if self._conf.my_custom_dataset:
                                try:
                                    if batch is None: continue
                                    anchor_inputs, candidate_inputs, levels, candidates_per_anchor = batch

                                    if self._conf.gpu:
                                        anchor_inputs = {k: v.cuda() for k, v in anchor_inputs.items()}
                                        candidate_inputs = {k: v.cuda() for k, v in candidate_inputs.items()}
                                        levels = levels.cuda()
                                    
                                    encoder = self._model.encoder
                                    loss_fn = self._model.pretraining_loss_layer

                                    anchor_outputs = encoder(input_ids=anchor_inputs['input_ids'], attention_mask=anchor_inputs['attention_mask'])[0]
                                    candidate_outputs = encoder(input_ids=candidate_inputs['input_ids'], attention_mask=candidate_inputs['attention_mask'])[0]

                                    anchor_mask_indices = (anchor_inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                                    candidate_mask_indices = (candidate_inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                                    anchor_embs = anchor_outputs[anchor_mask_indices[0], anchor_mask_indices[1]]
                                    candidate_embs = candidate_outputs[candidate_mask_indices[0], candidate_mask_indices[1]]
                                    
                                    total_loss = 0
                                    candidate_start_idx = 0
                                    for i in range(anchor_embs.shape[0]): # For each anchor in the batch
                                        num_cands = candidates_per_anchor[i]
                                        if num_cands == 0: continue
                                        candidate_end_idx = candidate_start_idx + num_cands
                                        
                                        loss_i = loss_fn(
                                            anchor_embs[i],
                                            candidate_embs[candidate_start_idx:candidate_end_idx],
                                            levels[candidate_start_idx:candidate_end_idx]
                                        )
                                        total_loss += loss_i
                                        candidate_start_idx = candidate_end_idx

                                    if anchor_embs.shape[0] > 0:
                                        loss = total_loss / anchor_embs.shape[0]
                                        losses.append(loss.item())
                                        loss.backward()

                                        if (iteration_idx + 1) % self._conf.grad_acc_iters == 0:
                                            if print_details: print("updating model parameters...")
                                            utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                            self._optimizer.step()
                                            self._optimizer.zero_grad()
                                            if print_details: print("OK")
                                        
                                        if print_details:
                                            print("Weighted Contrastive Loss: {:.4f}".format(loss.item()))
                                            print()
                                except Exception as e:
                                    print(f"Error in custom training loop: {e}")
                                    continue
                            # --- END OF NEW LOGIC BRANCH ---
                            
                            
                            
                            if (
                                    self._conf.nyt_dataset or
                                    self._conf.wikidata_dataset or
                                    self._conf.tacred_dataset or
                                    self._conf.retacred_dataset or
                                    self._conf.tacrev_dataset
                            ):
                                sent_input, \
                                    sent_ent_a_mask, \
                                    sent_ent_b_mask, \
                                    sent_prompt_mask, \
                                    sent_mlm_labels, \
                                    sent_attention_mask, \
                                    contrastive_labels = component_factory.ComponentFactory.create_nyt_pretrain_batch(
                                        batch,
                                        self._dataset.tokenizer.mask_token_id,
                                        self._dataset.tokenizer.pad_token_id,
                                        self._conf.mlm_percentage
                                    )
                                # Move tensors to GPU if specified.
                                if self._conf.gpu:
                                    sent_input = sent_input.cuda()
                                    sent_ent_a_mask = sent_ent_a_mask.cuda()
                                    sent_ent_b_mask = sent_ent_b_mask.cuda()
                                    sent_prompt_mask = sent_prompt_mask.cuda()
                                    if sent_mlm_labels is not None:
                                        sent_mlm_labels = sent_mlm_labels.cuda()
                                    sent_attention_mask = sent_attention_mask.cuda()
                                    contrastive_labels = contrastive_labels.cuda()

                                # Pass to the model
                                try:
                                    loss_dict = self._model(
                                        sent_a_input=sent_input,
                                        sent_a_ent_a_mask=sent_ent_a_mask,
                                        sent_a_ent_b_mask=sent_ent_b_mask,
                                        sent_a_prompt_mask=sent_prompt_mask,
                                        sent_a_mlm_labels=sent_mlm_labels,
                                        sent_a_attention_mask=sent_attention_mask,
                                        contrastive_labels=contrastive_labels
                                    )
                                    loss = None
                                    if self._conf.ht and self._conf.rel_mask:
                                        loss = torch.mean(
                                            torch.cat(
                                                (
                                                    loss_dict["h_and_t"].view(-1),
                                                    loss_dict["rel_mask"].view(-1)
                                                )
                                            ),
                                            dim=0
                                        )
                                    elif self._conf.ht:
                                        loss = loss_dict["h_and_t"].view(-1)

                                    elif self._conf.rel_mask:
                                        loss = loss_dict["rel_mask"].view(-1)
                                    else:
                                        loss = loss_dict["all"].view(-1)

                                    loss.backward()
                                    losses.append(loss.item())
                                    if (
                                            iteration_idx + 1) % self._conf.grad_acc_iters == 0:  # -> all needed grads accumulated

                                        if print_details:
                                            print("updating model parameters...")

                                        utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                        self._optimizer.step()
                                        self._optimizer.zero_grad()

                                        if print_details:
                                            print("OK")

                                    if print_details:
                                        print("H and T contrastive loss: {:.4f}".format(loss_dict["h_and_t"].item()))
                                        print("Rel mask contrastive loss: {:.4f}".format(loss_dict["rel_mask"].item()))
                                        print("H, T, and rel mask contrastive loss: {:.4f}".format(
                                            loss_dict["all"].item()))
                                        print()
                                        print("Avg. loss: {:.4f}".format(loss.item()))
                                        print("Ok")
                                except Exception as e:
                                    continue
                                if print_details:
                                    print()
                                    util.printing_tools.print_end("Iteration {}".format(iteration_idx), level=1)

                            else:
                                sent_a_input, sent_a_ent_a_mask, sent_a_ent_b_mask, sent_a_prompt_mask, sent_a_mlm_labels, sent_a_attention_mask, sent_b_input, sent_b_ent_a_mask, sent_b_ent_b_mask, sent_b_prompt_mask, sent_b_mlm_labels, sent_b_attention_mask, contrastive_labels = component_factory.ComponentFactory.create_pretrain_batch(
                                    batch, self._dataset.tokenizer.mask_token_id, self._dataset.tokenizer.pad_token_id,
                                    self._conf.mlm_percentage)

                                # Move tensors to GPU if specified.
                                if self._conf.gpu:
                                    sent_a_input = sent_a_input.cuda()
                                    sent_a_ent_a_mask = sent_a_ent_a_mask.cuda()
                                    sent_a_ent_b_mask = sent_a_ent_b_mask.cuda()
                                    sent_a_prompt_mask = sent_a_prompt_mask.cuda()
                                    if sent_a_mlm_labels is not None:
                                        sent_a_mlm_labels = sent_a_mlm_labels.cuda()
                                    sent_a_attention_mask = sent_a_attention_mask.cuda()
                                    sent_b_input = sent_b_input.cuda()
                                    sent_b_ent_a_mask = sent_b_ent_a_mask.cuda()
                                    sent_b_ent_b_mask = sent_b_ent_b_mask.cuda()
                                    sent_b_prompt_mask = sent_b_prompt_mask.cuda()
                                    if sent_b_mlm_labels is not None:
                                        sent_b_mlm_labels = sent_b_mlm_labels.cuda()
                                    sent_b_attention_mask = sent_b_attention_mask.cuda()
                                    contrastive_labels = contrastive_labels.cuda()

                                # Pass to the model
                                try:
                                    loss_dict = self._model(
                                        sent_a_input,
                                        sent_a_ent_a_mask,
                                        sent_a_ent_b_mask,
                                        sent_a_prompt_mask,
                                        sent_a_mlm_labels,
                                        sent_a_attention_mask,
                                        contrastive_labels,
                                        sent_b_input,
                                        sent_b_ent_a_mask,
                                        sent_b_ent_b_mask,
                                        sent_b_prompt_mask,
                                        sent_b_mlm_labels,
                                        sent_b_attention_mask
                                    )
                                    loss = None
                                    if self._conf.ht and self._conf.rel_mask:
                                        loss = torch.mean(
                                            torch.cat(
                                                (
                                                    loss_dict["h_and_t"].view(-1),
                                                    loss_dict["rel_mask"].view(-1)
                                                )
                                            ),
                                            dim=0
                                        )
                                    elif self._conf.ht:
                                        loss = loss_dict["h_and_t"].view(-1)

                                    elif self._conf.rel_mask:
                                        loss = loss_dict["rel_mask"].view(-1)
                                    else:
                                        loss = loss_dict["all"].view(-1)

                                    loss.backward()
                                    losses.append(loss.item())
                                    if (
                                            iteration_idx + 1) % self._conf.grad_acc_iters == 0:  # -> all needed grads accumulated

                                        if print_details:
                                            print("updating model parameters...")

                                        utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                        self._optimizer.step()
                                        self._optimizer.zero_grad()

                                        if print_details:
                                            print("OK")

                                    if print_details:
                                        print("H and T contrastive loss: {:.4f}".format(loss_dict["h_and_t"].item()))
                                        print("Rel mask contrastive loss: {:.4f}".format(loss_dict["rel_mask"].item()))
                                        print("H, T, and rel mask contrastive loss: {:.4f}".format(
                                            loss_dict["all"].item()))
                                        print()
                                        print("Avg. loss: {:.4f}".format(loss.item()))
                                        print("Ok")
                                except Exception as e:
                                    continue
                                if print_details:
                                    print()
                                    util.printing_tools.print_end("Iteration {}".format(iteration_idx), level=1)

                        else:
                            try:
                                input_seq, \
                                    head_entity_mask, \
                                    tail_entity_mask, \
                                    relation_mask, \
                                    relation_types, \
                                    mlm_labels = component_factory.ComponentFactory.create_batch(
                                    batch,
                                    self._dataset.tokenizer.mask_token_id,
                                    self._dataset.tokenizer.pad_token_id,
                                    self._conf.mlm_percentage
                                )

                                # Move tensors to gpu is specified
                                if self._conf.gpu:
                                    input_seq = input_seq.cuda()
                                    head_entity_mask = head_entity_mask.cuda()
                                    tail_entity_mask = tail_entity_mask.cuda()
                                    relation_mask = relation_mask.cuda()
                                    relation_types = relation_types.cuda()
                                    mlm_labels = mlm_labels.cuda()
                                model_output = self._model(
                                    input_seq=input_seq,
                                    head_entity_mask=head_entity_mask,
                                    tail_entity_mask=tail_entity_mask,
                                    relation_mask=relation_mask,
                                    mlm_labels=mlm_labels
                                )
                                classification_loss = 0
                                mlm_loss = 0

                                classification_loss = self._cross_entropy_loss(
                                    model_output["classfication_preds"].squeeze(1),
                                    relation_types.view(-1)
                                )
                                if model_output["mlm_loss"] is not None:
                                    mlm_loss = model_output["mlm_loss"]
                                if not self._conf.freeze_rel_enc:
                                    loss = classification_loss + mlm_loss
                                else:
                                    loss = classification_loss

                                loss.backward()
                                if (
                                        iteration_idx + 1) % self._conf.grad_acc_iters == 0:  # -> all needed grads accumulated

                                    if print_details:
                                        print("updating model parameters...")

                                    utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                    self._optimizer.step()
                                    self._optimizer.zero_grad()

                                    if print_details:
                                        print("OK")

                                if print_details:
                                    print("Classification loss: {:.4f}".format(classification_loss))
                                    if not self._conf.freeze_rel_enc:
                                        print("MLM loss: {:.4f}".format(mlm_loss))
                                    print()
                                    print("Avg. loss: {:.4f}".format(loss))
                                    print("Ok")
                            except Exception as e:
                                continue
                            if print_details:
                                print()
                                util.printing_tools.print_end("Iteration {}".format(iteration_idx), level=1)
                            # End of iteration.
                    # Store iteration duration and update step counter
                    iteration_durations.append(iteration_timer.total)
                    num_steps += 1
            # Print additional epoch details
            print("# of iterations: {}".format(len(iteration_durations)))
            print("Avg. duration per iteration: {:.3f}s".format(np.mean(iteration_durations)))
            print()
            util.printing_tools.print_end("Epoch {}".format(epoch), level=0)

            # End of epoch

            # Store epoch duration
            epoch_durations.append(epoch_timer.total)

            # Create checkpoint
            print("Creating checkpoint...")
            ckpt = None
            if self._conf.pretrain:
                ckpt = rel_encoder_checkpoint.RelEncoderCheckpoint(
                    epoch,
                    self._model.encoder.state_dict(),
                    self._model.pretraining_loss_layer.state_dict(),
                    self._optimizer.state_dict
                )
                ckpt.average_loss = np.mean(losses)
            else:
                ckpt = encore_model_checkpoint.EnCoreModelCheckpoint(
                    epoch,
                    self._model.state_dict(),
                    self._optimizer.state_dict()
                )
            ckpt_path = self._ckpt_saver.save(ckpt, steps=epoch)
            self._deliver_ckpt(ckpt_path)
            print("OK")
            print()

        print("Avg. duration per epoch: {:3f}s".format(np.mean(epoch_durations)))
