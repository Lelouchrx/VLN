import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLProcessor
from vln.utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class VLNModel(Qwen3VLModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VLNModel, self).__init__(config)
        

class VLNForCausalLM(Qwen3VLForConditionalGeneration):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Qwen3VLForConditionalGeneration, self).__init__(config)

        self.model = VLNModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model
    
    
    
    
    
    
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     return_dict: Optional[bool] = None,
    #     **kwargs
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     return super().forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         labels=labels,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         pixel_values=pixel_values,
    #         image_grid_thw=image_grid_thw,
    #         return_dict=return_dict
    #     )
    
    # @torch.no_grad()
    # def generate(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     image_grid_thw: Optional[torch.Tensor] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     return super().generate(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         image_grid_thw=image_grid_thw,
    #         **kwargs
    #     )
    