from __future__ import annotations

import torch
import torch.nn as nn
import json
from overrides import overrides
from typing import Optional, List
from collections import OrderedDict
from transformers import DebertaV2Model, DebertaV2Config
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutput
from src.utils.utils import Task, Tasks
from torch.nn import LayerNorm


class EmbeddingOutputs(OrderedDict):

    embeddings: torch.FloatTensor = None
    alpha: torch.FloatTensor = None
    beta: torch.FloatTensor = None, 
    annotator_embed_before_alpha: torch.FloatTensor = None
    annotator_embed_after_alpha: torch.FloatTensor = None
    annotation_embed_before_beta: torch.FloatTensor = None
    annotation_embed_after_beta: torch.FloatTensor = None
    sentence_embed: torch.FloatTensor = None

# Modified based on the DeBERTa embedding from the library
# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm
class CustomizedDebertaV2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, num_annotators, label_nums, \
                 broadcast_annotator_embedding, broadcast_annotation_embedding,
                 include_pad_annotation, method):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.sent_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.annotation_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.annotator_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.annotator_embed = nn.Embedding(num_annotators, config.hidden_size)
        # consider the padding, we have label_nums + 1 possible annotations
        self.annotation_embed = nn.Embedding(label_nums + 1, config.hidden_size, padding_idx=0)    
        self.broadcast_annotator_embedding = broadcast_annotator_embedding
        self.broadcast_annotation_embedding = broadcast_annotation_embedding
        self.include_pad_annotation = include_pad_annotation
        self.method = method
    
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

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None, 
        annotator_ids: Optional[torch.LongTensor] = None,
        annotations: Optional[torch.LongTensor] = None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
        
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
                    annotation_embed_after_beta = beta * (torch.div(self.annotation_embed(annotations).sum(dim=1), AN))
                    if self.broadcast_annotation_embedding:
                        embeddings += annotation_embed_after_beta.unsqueeze(1).repeat(1, K, 1)
                    else:
                        embeddings = embeddings.clone()  # Clone the embeddings tensor
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
                    annotation_embed_after_beta = beta * torch.div(annotation_sum, mask_sum)
                    if self.broadcast_annotation_embedding:
                        embeddings += annotation_embed_after_beta.unsqueeze(1).repeat(1, K, 1)
                    else:
                        embeddings = embeddings.clone()
                        embeddings[:, 0, :] += annotation_embed_after_beta
        elif self.method == "concat":
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

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return EmbeddingOutputs(embeddings=embeddings, 
                            alpha=alpha, 
                            beta=beta, 
                            annotator_embed_before_alpha=annotator_embed_before_alpha,
                            annotator_embed_after_alpha=annotator_embed_after_alpha, 
                            annotation_embed_before_beta=annotation_embed_before_beta,
                            annotation_embed_after_beta=annotation_embed_after_beta, 
                            sentence_embed=sentence_embed)



# All copied from the original
# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True

# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout

# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

# Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification with Deberta->DebertaV2
class DebertaV2MultiChoice(DebertaV2Model):
    def __init__(self, config: DebertaV2Config, tasks: Tasks, decoder_tokenizers: dict,
            num_annotators: int, label_nums: int , broadcast_annotator_embedding: bool,
            broadcast_annotation_embedding: bool, annotator_id_path: str,
            use_annotator_embed: bool, use_annotation_embed: bool,
            include_pad_annotation: bool, method: str) -> None:  # noqa):
        
        super().__init__(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.tasks = tasks
        self.decoder_tokenizers = decoder_tokenizers
        self.classifiers = nn.ModuleDict({})
        for task in self.tasks:
            # ModuleDict explicitly asks the key to be of type str
            self.classifiers.update({task.name: nn.Linear(config.hidden_size, self.decoder_tokenizers[task].num_labels)})

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

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

        with open(self.annotator_id_path, 'r') as f:
            annotator_ids = json.load(f)
        self.annotator_mapping = {annotator_id: i for i, annotator_id in enumerate(annotator_ids.keys())}
        self.embeddings = CustomizedDebertaV2Embeddings(config, 
                    num_annotators = self.num_annotators, 
                    label_nums = self.label_nums, 
                    broadcast_annotator_embedding = self.broadcast_annotator_embedding, 
                    broadcast_annotation_embedding = self.broadcast_annotation_embedding,
                    include_pad_annotation=self.include_pad_annotation,
                    method=self.method)

        # Initialize weights and apply final processing
        self.post_init()

    def super_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        annotator_ids=None,
        annotations=None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
            annotator_ids=annotator_ids,
            annotations=annotations
        )

        encoder_outputs = self.encoder(
            output["embeddings"],
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(output["embeddings"])
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        output["alpha"] = output["alpha"].squeeze() if output["alpha"] is not None and output["alpha"][0] is not None else None
        output["beta"] = output["beta"].squeeze() if output["beta"] is not None and output["beta"][0] is not None else None

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
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
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        return self.classifiers[task.name](pooled_output), embedding_output
    