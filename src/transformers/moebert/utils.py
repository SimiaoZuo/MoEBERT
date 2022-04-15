import numpy as np
import pickle
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple


def use_experts(layer_idx):
    return True


def process_ffn(model):
    if model.config.model_type == "bert":
        inner_model = model.bert
    else:
        raise ValueError("Model type not recognized.")

    for i in range(model.config.num_hidden_layers):
        model_layer = inner_model.encoder.layer[i]
        if model_layer.use_experts:
            model_layer.importance_processor.load_experts(model_layer)


class ImportanceProcessor:
    GROUP_RANDOM_SATE = 10000  # for expert initialization

    def __init__(self, config, layer_idx, num_local_experts, local_group_rank):
        self.num_experts = config.moebert_expert_num  # total number of experts
        self.num_local_experts = num_local_experts  # number of experts on this device
        self.local_group_rank = local_group_rank  # rank in the current process group
        self.intermediate_size = config.moebert_expert_dim  # FFN hidden dimension
        self.share_importance = config.moebert_share_importance  # number of shared FFN dimension

        importance = ImportanceProcessor.load_importance_single(config.moebert_load_importance)[layer_idx, :]
        # importance = np.random.permutation(self.intermediate_size * self.num_experts)
        self.importance = self._split_importance(importance)

        self.is_moe = False  # safety check

    @staticmethod
    def load_importance_single(importance_files):
        with open(importance_files, "rb") as file:
            data = pickle.load(file)
            data = data["idx"]
        return np.array(data)

    def _split_importance(self, arr, random_selection=False):
        result = []
        top_importance = arr[:self.share_importance]
        remain = arr[self.share_importance:]
        all_experts_remain = []
        for i in range(self.num_experts):
            all_experts_remain.append(remain[i::self.num_experts])
            # all_experts_remain.append(remain[i * self.num_experts:(i + 1) * self.num_experts])
        all_experts_remain = np.array(all_experts_remain)

        for i in range(self.num_local_experts):
            if random_selection:
                temp = np.random.RandomState(self.GROUP_RANDOM_SATE * self.local_group_rank + i).permutation(remain)
            else:
                temp = all_experts_remain[self.num_local_experts * self.local_group_rank + i]
            temp = np.concatenate((top_importance, temp))
            temp = temp[:self.intermediate_size]
            result.append(temp.copy())
        result = np.array(result)
        return result

    def load_experts(self, model_layer):
        expert_list = model_layer.experts.experts
        fc1_weight_data = model_layer.intermediate.dense.weight.data
        fc1_bias_data = model_layer.intermediate.dense.bias.data
        fc2_weight_data = model_layer.output.dense.weight.data
        fc2_bias_data = model_layer.output.dense.bias.data
        layernorm_weight_data = model_layer.output.LayerNorm.weight.data
        layernorm_bias_data = model_layer.output.LayerNorm.bias.data
        for i in range(self.num_local_experts):
            idx = self.importance[i]
            expert_list[i].fc1.weight.data = fc1_weight_data[idx, :].clone()
            expert_list[i].fc1.bias.data = fc1_bias_data[idx].clone()
            expert_list[i].fc2.weight.data = fc2_weight_data[:, idx].clone()
            expert_list[i].fc2.bias.data = fc2_bias_data.clone()
            expert_list[i].LayerNorm.weight.data = layernorm_weight_data.clone()
            expert_list[i].LayerNorm.bias.data = layernorm_bias_data.clone()
        del model_layer.intermediate
        del model_layer.output
        self.is_moe = True


class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size, dropout):
        nn.Module.__init__(self)

        # first layer
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # second layer
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor):
        input_tensor = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@dataclass
class MoEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None


@dataclass
class MoEModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None
