from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig, ElectraForMaskedLM
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions, ElectraForPreTrainingOutput

### Components ### 

class ElectraOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None:
        super(ElectraOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class ElectraDescriminatorHead(nn.Module):
    def __init__(self, config : ElectraConfig) -> None:
        super().__init__()
        self.config = config
        self.descriminator_predictions = ElectraDiscriminatorPredictions(config)
    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

### Models ###

class ElectraForNSP(ElectraPreTrainedModel):
    def __init__(self, generator_config: ElectraConfig, **kwargs) -> None:
        super().__init__(generator_config)
        self.generator = ElectraModel(generator_config)
        self.NSP = ElectraOnlyNSPHead(generator_config)
        self.post_init()

class ElectraForDiscrim(ElectraPreTrainedModel):
    def __init__(self, generator_config: ElectraConfig, descriminator_config: ElectraConfig, tie_weights=True, **kwargs) -> None:
        super().__init__(descriminator_config)
        self.generator = ElectraForMaskedLM(generator_config)
        self.electra = ElectraModel(descriminator_config)
        self.descriminator = ElectraDescriminatorHead(descriminator_config)

        self.post_init()

        if tie_weights:
            self.discriminator.electra.embeddings = self.generator.electra.embeddings
            self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:
        
        
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
        self.descriminator = ElectraDescriminatorHead(descriminator_config)
        self.NSP = ElectraOnlyNSPHead(descriminator_config)
        self.post_init()

        if tie_weights:
            self.electra.embeddings = self.generator.electra.embeddings
            self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight