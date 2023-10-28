from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig, ElectraForMaskedLM
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions, ElectraForPreTrainingOutput

from .loss import NSPLoss, MLMLoss

### Components ### 

nspLoss = NSPLoss()
mlmLoss = MLMLoss()

class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ElectraOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None:
        super(ElectraOnlyNSPHead, self).__init__()
        self.cls = ElectraPooler(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = self.cls(hidden_states)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class ElectraDiscriminatorHead(nn.Module):
    def __init__(self, config : ElectraConfig) -> None:
        super().__init__()
        self.descriminator_predictions = ElectraDiscriminatorPredictions(config)
    
    def forward(self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        discriminator_sequence_output = hidden_states[0]
        return self.discriminator_predictions(discriminator_sequence_output)


### Models ###

class ElectraForNSP(ElectraPreTrainedModel):
    def __init__(self, generator_config: ElectraConfig, **kwargs) -> None:
        super().__init__(generator_config)
        self.electra = ElectraModel(generator_config)
        self.NSP = ElectraOnlyNSPHead(generator_config)
        self.post_init()
    
    def forward(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        **kwargs) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.NSP(generator_hidden_states.hidden_states)

        loss = nspLoss(logits, next_sentence_label)

        if not return_dict:
            output = (logits,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output 
        
        return ElectraForPreTrainingOutput(
            loss=nspLoss(logits, next_sentence_label),
            logits=logits,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )

class ElectraForDiscrim(ElectraPreTrainedModel):
    def __init__(self, generator_config: ElectraConfig, discriminator_config: ElectraConfig, tie_weights=True, **kwargs) -> None:
        super().__init__(discriminator_config)
        self.electra = ElectraModel(discriminator_config)
        self.generator = ElectraForMaskedLM(generator_config)
        self.discriminator = ElectraDiscriminatorHead(discriminator_config)
        self.post_init()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)

        if tie_weights:
            self.electra.embeddings = self.generator.electra.embeddings
            self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
    
    def sample(self, logits):
        "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
        gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
        return (logits.float() + gumbel).argmax(dim=-1)
    
    def forward(self, input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            mlm_labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:

        # Generator Model
        generator_hidden_states = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Determine which tokens have MLM applied
        mlm_applied = mlm_labels != -100
        mlm_generator_logits = generator_hidden_states.logits

        # Sampling and preparing inputs for the discriminator
        with torch.no_grad():
            pred_toks = self.sample(mlm_generator_logits[mlm_applied])  # ( #mlm_positions, )
            generated = input_ids.clone()
            generated[mlm_applied] = pred_toks
            is_replaced = mlm_labels.clone()
            is_replaced[mlm_applied] = (pred_toks != mlm_labels[mlm_applied])

        # Discriminator Model
        discriminator_hidden_states = self.electra(
            input_ids=generated,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the discriminator logits
        logits = self.discriminator(discriminator_hidden_states.hidden_states)

        # Calculate the loss using the custom loss function
        loss = mlmLoss((mlm_generator_logits[mlm_applied], logits, is_replaced, attention_mask[mlm_applied], mlm_applied), mlm_labels[mlm_applied])
        
        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class ElectraForMultiObj(ElectraPreTrainedModel):
    def __init__(self, generator_config: ElectraConfig, descriminator_config: ElectraConfig, tie_weights=True, **kwargs) -> None:
        super().__init__(descriminator_config)
        self.electra = ElectraModel(descriminator_config)
        self.generator = ElectraForMaskedLM(generator_config)
        self.descriminator = ElectraDiscriminatorHead(descriminator_config)
        self.NSP = ElectraOnlyNSPHead(descriminator_config)
        self.post_init()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)

        if tie_weights:
            self.electra.embeddings = self.generator.electra.embeddings
            self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
    
    def forward(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        nsp_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        **kwargs) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:

        # Generator Model
        generator_hidden_states = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Determine which tokens have MLM applied
        mlm_applied = mlm_labels != -100
        mlm_generator_logits = generator_hidden_states.logits

        # Sampling and preparing inputs for the discriminator
        with torch.no_grad():
            pred_toks = self.sample(mlm_generator_logits[mlm_applied])  # ( #mlm_positions, )
            generated = input_ids.clone()
            generated[mlm_applied] = pred_toks
            is_replaced = mlm_labels.clone()
            is_replaced[mlm_applied] = (pred_toks != mlm_labels[mlm_applied])

        # Discriminator Model
        discriminator_hidden_states = self.electra(
            input_ids=generated,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_logits = self.discriminator(discriminator_hidden_states.hidden_states)
        nsp_logits = self.NSP(discriminator_hidden_states.hidden_states)

        # Get logits
        
        mlm_loss = mlmLoss((mlm_generator_logits[mlm_applied], mlm_logits, is_replaced, attention_mask[mlm_applied], mlm_applied), mlm_labels[mlm_applied])
        nsp_loss = nspLoss(nsp_logits, nsp_labels)

        loss = mlm_loss + nsp_loss

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=mlm_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )