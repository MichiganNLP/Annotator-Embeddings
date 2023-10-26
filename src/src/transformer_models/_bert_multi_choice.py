from __future__ import annotations

import torch
import torch.nn as nn
import json
from overrides import overrides
from typing import Optional, List
from collections import OrderedDict
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from src.utils.utils import Task, Tasks


class EmbeddingOutputs(OrderedDict):

    embeddings: torch.FloatTensor = None
    alpha: torch.FloatTensor = None
    beta: torch.FloatTensor = None, 
    annotator_embed_before_alpha: torch.FloatTensor = None
    annotator_embed_after_alpha: torch.FloatTensor = None
    annotation_embed_before_beta: torch.FloatTensor = None
    annotation_embed_after_beta: torch.FloatTensor = None
    sentence_embed: torch.FloatTensor = None


class CustomizedBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, num_annotators, label_nums, \
                 broadcast_annotator_embedding, broadcast_annotation_embedding,
                 include_pad_annotation, method, embed_wo_weight):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.sent_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.annotation_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.annotator_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.annotator_embed = nn.Embedding(num_annotators, config.hidden_size)
        # consider the padding, we have label_nums + 1 possible annotations
        self.annotation_embed = nn.Embedding(label_nums + 1, config.hidden_size, padding_idx=0)    
        self.broadcast_annotator_embedding = broadcast_annotator_embedding
        self.broadcast_annotation_embedding = broadcast_annotation_embedding
        self.include_pad_annotation = include_pad_annotation
        self.method = method
        self.embed_wo_weight = embed_wo_weight

    def _calculate_sent_alpha(self, embeddings):
        # calculate the sentence embedding based on the average of the inputs_embeds
        # we include the position embedding as well as the type embeddings here
        # as those are also valid information to be considered
        sent_embeds = torch.mean(embeddings, dim=1, keepdim=True)
        sent_embeds = torch.transpose(sent_embeds, 1, 2)
        alpha_sent = torch.einsum('ji, kim->kjm', self.sent_W, sent_embeds)
        return alpha_sent
    
    def _calculate_annotator_alpha(self, ann):
        antr_embed = torch.transpose(self.annotator_embed(ann).unsqueeze(1), 1, 2)
        alpha_ant = torch.einsum('ji, kim->kjm', self.annotator_W, antr_embed)
        return alpha_ant
    
    def _calculate_annotation_alpha(self, ann):
        ann_embed = torch.mean(self.annotation_embed(ann), dim=1, keepdim=True)
        ann_embed = torch.transpose(ann_embed, 1, 2)
        alpha_ant = torch.einsum('ji, kim->kjm', self.annotation_W, ann_embed)
        return alpha_ant

    def _calculate_alpha(self, alpha_sent, alpha_ant):
        alpha = torch.einsum('bxm,bxn->bmn', alpha_sent, alpha_ant)
        alpha = torch.squeeze(alpha, dim=2) # B * 1
        return alpha

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        annotator_ids: Optional[torch.LongTensor] = None,
        annotations: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        alpha_sent = self._calculate_sent_alpha(embeddings)

        alpha = None   # annotator weight
        beta = None     # annotation weight

        annotator_embed_before_alpha = None  # annotator embedding
        annotation_embed_before_beta = None # annotation embedding
        annotator_embed_after_alpha = None  # annotator embedding
        annotation_embed_after_beta = None # annotation embedding
        sentence_embed = embeddings # sentence embedding

        if self.method == "add":
            # 101 is the [CLS] token, we experiment with adding the annotator embedding to [CLS]
            _, K, _ = embeddings.shape
            if annotator_ids is not None:
                alpha_annotator = self._calculate_annotator_alpha(annotator_ids)
                alpha = self._calculate_alpha(alpha_sent, alpha_annotator)
                annotator_embed_before_alpha = self.annotator_embed(annotator_ids)
                if self.embed_wo_weight:
                    annotator_embed_after_alpha = annotator_embed_before_alpha
                else:
                    annotator_embed_after_alpha = alpha * self.annotator_embed(annotator_ids)
                if self.broadcast_annotator_embedding:
                    embeddings += annotator_embed_after_alpha.unsqueeze(1).repeat(1, K, 1)
                else:
                    embeddings = embeddings.clone()
                    embeddings[:, 0, :] += annotator_embed_after_alpha
            if annotations is not None:
                # does not ignore the 0 padding value; incorrect
                # embeddings[:, 0, :] += torch.mean(self.annotation_embed(annotations), dim=1)
                # NOTE: there is a case if there is no annotations
                
                alpha_annotations = self._calculate_annotation_alpha(annotations)
                beta = self._calculate_alpha(alpha_sent, alpha_annotations)

                if self.include_pad_annotation:
                    _, AN = annotations.shape
                    # padding will always be 0 in the embedding, we ignore the padding item when calculate the mean
                    annotation_embed_before_beta = torch.div(self.annotation_embed(annotations).sum(dim=1), AN)
                    if self.embed_wo_weight:
                        annotation_embed_after_beta = annotation_embed_before_beta
                    else:
                        annotation_embed_after_beta = beta * (torch.div(self.annotation_embed(annotations).sum(dim=1), AN))
                    if self.broadcast_annotation_embedding:
                        embeddings += annotation_embed_after_beta.unsqueeze(1).repeat(1, K, 1)
                    else:
                        embeddings = embeddings.clone()
                        embeddings[:, 0, :] += annotation_embed_after_beta
                else:
                    # not include the padding
                    mask = annotations !=0  # batch_size * other annotation size
                    ann_embed = self.annotation_embed(annotations)
                    _, _, D = ann_embed.shape
                    mask_expand = mask.unsqueeze(2).repeat(1, 1, D)
                    assert mask_expand.shape == ann_embed.shape
                    masked_ann_embed = mask_expand * ann_embed  # element-wise multiplication
                    annotation_sum = masked_ann_embed.sum(dim=1)

                    mask_sum = mask.sum(dim=1).unsqueeze(dim=1)
                    # Replace NaN values with zeros
                    row_has_zero_mask_sum = mask_sum == 0   # B * 1
                    mask_sum[row_has_zero_mask_sum] = 1
                    # padding will always be 0 in the embedding, we ignore the padding item when calculate the mean
                    annotation_embed_before_beta = torch.div(annotation_sum, mask_sum)
                    if self.embed_wo_weight:
                        annotation_embed_after_beta = annotation_embed_before_beta
                    else:
                        annotation_embed_after_beta = beta * torch.div(annotation_sum, mask_sum)
                    if self.broadcast_annotation_embedding:
                        embeddings += annotation_embed_after_beta.unsqueeze(1).repeat(1, K, 1)
                    else:
                        embeddings = embeddings.clone()
                        embeddings[:, 0, :] += annotation_embed_after_beta
        elif self.method == "concat":
            # NOTE: for the ablation of without weight, we only conduct experiments on add method
            _, K, _ = embeddings.shape

            if annotator_ids is not None:
                alpha_annotator = self._calculate_annotator_alpha(annotator_ids)
                alpha = self._calculate_alpha(alpha_sent, alpha_annotator)
                # cannot broadcast in the concatenation case
                assert self.broadcast_annotator_embedding is False
                annotator_embed_before_alpha = self.annotator_embed(annotator_ids)
                annotator_embed_after_alpha = alpha *  self.annotator_embed(annotator_ids)
                annotator_embed = annotator_embed_after_alpha
                embeddings = torch.cat((embeddings[:, :1, :], annotator_embed.unsqueeze(1), embeddings[:, 1:, :]), dim=1)

            if annotations is not None:
                # embeddings[:, 0, :] += torch.mean(self.annotation_embed(annotations), dim=1)
                # NOTE: there is a case if there is no annotations
                alpha_annotations = self._calculate_annotation_alpha(annotations)
                beta = self._calculate_alpha(alpha_sent, alpha_annotations)
                annotation_embed = None
                if self.include_pad_annotation:
                    _, AN = annotations.shape
                    # padding will always be 0 in the embedding, we ignore the padding item when calculate the mean
                    annotation_embed_before_beta = torch.div(self.annotation_embed(annotations).sum(dim=1), AN)
                    annotation_embed_after_beta = beta * (torch.div(self.annotation_embed(annotations).sum(dim=1), AN))
                    annotation_embed = annotation_embed_after_beta
                else:
                    # not include the padding
                    mask = annotations !=0  # batch_size * other annotation size
                    ann_embed = self.annotation_embed(annotations)
                    _, _, D = ann_embed.shape
                    mask_expand = mask.unsqueeze(2).repeat(1, 1, D)
                    assert mask_expand.shape == ann_embed.shape
                    masked_ann_embed = mask_expand * ann_embed  # element-wise multiplication
                    annotation_sum = masked_ann_embed.sum(dim=1)

                    mask_sum = mask.sum(dim=1).unsqueeze(dim=1)
                    # Replace NaN values with zeros
                    row_has_zero_mask_sum = mask_sum == 0   # B * 1
                    mask_sum[row_has_zero_mask_sum] = 1
                    # padding will always be 0 in the embedding, we ignore the padding item when calculate the mean
                    annotation_embed_before_beta = torch.div(annotation_sum, mask_sum)
                    annotation_embed_after_beta = beta * (torch.div(annotation_sum, mask_sum))
                    annotation_embed = annotation_embed_after_beta
                if annotator_ids is None:
                    # position at 1:
                    embeddings = torch.cat((embeddings[:, :1, :], annotation_embed.unsqueeze(1), embeddings[:, 1:, :]), dim=1)
                else:
                    # position at 2:
                    embeddings = torch.cat((embeddings[:, :2, :], annotation_embed.unsqueeze(1), embeddings[:, 2:, :]), dim=1)
        else:
            raise RuntimeError(f"Method {self.method} not supported!")
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return EmbeddingOutputs(embeddings=embeddings, 
                                alpha=alpha, 
                                beta=beta, 
                                annotator_embed_before_alpha=annotator_embed_before_alpha,
                                annotator_embed_after_alpha=annotator_embed_after_alpha, 
                                annotation_embed_before_beta=annotation_embed_before_beta,
                                annotation_embed_after_beta=annotation_embed_after_beta, 
                                sentence_embed=sentence_embed)

class BERTMultiChoice(BertModel):

    def __init__(self, config: BertConfig, tasks: Tasks, decoder_tokenizers: dict,
            num_annotators: int, label_nums: int , broadcast_annotator_embedding: bool,
            broadcast_annotation_embedding: bool, annotator_id_path: str,
            use_annotator_embed: bool, use_annotation_embed: bool,
            include_pad_annotation: bool, method: str,
            embed_wo_weight: bool) -> None:  # noqa
        super().__init__(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)   # randomly zero out some input elements, no need to be shared
        self.tasks = tasks
        self.decoder_tokenizers = decoder_tokenizers
        self.classifiers = nn.ModuleDict({})
        for task in self.tasks:
            # ModuleDict explicitly asks the key to be of type str
            self.classifiers.update({task.name: nn.Linear(config.hidden_size, self.decoder_tokenizers[task].num_labels)})

        # experiment purpose
        self.num_annotators = num_annotators
        self.label_nums = label_nums
        self.broadcast_annotator_embedding = broadcast_annotator_embedding
        self.broadcast_annotation_embedding = broadcast_annotation_embedding
        self.annotator_id_path = annotator_id_path
        self.use_annotator_embed = use_annotator_embed
        self.use_annotation_embed = use_annotation_embed
        self.include_pad_annotation = include_pad_annotation
        self.method = method
        self.embed_wo_weight = embed_wo_weight

        with open(self.annotator_id_path, 'r') as f:
            annotator_ids = json.load(f)
        self.annotator_mapping = {annotator_id: i for i, annotator_id in enumerate(annotator_ids.keys())}
        self.embeddings = CustomizedBertEmbeddings(config, 
                    num_annotators = self.num_annotators, 
                    label_nums = self.label_nums, 
                    broadcast_annotator_embedding = self.broadcast_annotator_embedding, 
                    broadcast_annotation_embedding = self.broadcast_annotation_embedding,
                    include_pad_annotation=self.include_pad_annotation,
                    method=self.method,
                    embed_wo_weight=self.embed_wo_weight)
    
    def super_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        annotator_ids=None,
        annotations=None,
        **kwargs
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            annotator_ids=annotator_ids,
            annotations=annotations
        )
        encoder_outputs = self.encoder(
            output["embeddings"],
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        output["alpha"] = output["alpha"].squeeze() if output["alpha"] is not None and output["alpha"][0] is not None else None
        output["beta"] = output["beta"].squeeze() if output["beta"] is not None and output["beta"][0] is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), output


    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, \
        task: Task | None = None, annotator_ids=None, annotations=None, **kwargs) -> Any:
        if (not self.use_annotation_embed) or (kwargs.get("disable_annotator_ids", False)) \
            or (annotations is not None and all(ann.numel() == 0 for ann in annotations)):
            # or there is no annotations simply
            annotations = None
        if not self.use_annotator_embed or (kwargs.get("disable_annotation", False)):
            annotator_ids = None

        if kwargs.get("disable_question", False):
            # we disable question (only leave the [CLS] token)
            input_ids = input_ids[:, 0:1]
            attention_mask = attention_mask[:, 0:1]

        assert self.method in ["add", "concat"]
        if self.method == "concat":
            # we need to move the attention mask
            B, _ = attention_mask.shape
            if annotator_ids is not None:
                attention_mask = torch.cat((attention_mask[:, :1], torch.ones((B, 1), device=self.device), attention_mask[:, 1:]), dim=1)
            if annotations is not None:
                attention_mask = torch.cat((attention_mask[:, :1], torch.ones((B, 1), device=self.device), attention_mask[:, 1:]), dim=1)
        outputs, embedding_output = self.super_forward(input_ids, attention_mask=attention_mask, annotator_ids=annotator_ids, annotations=annotations, **kwargs)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        return self.classifiers[task.name](sequence_output), embedding_output
