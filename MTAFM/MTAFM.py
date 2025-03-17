from math import sqrt
from typing import List
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ImPLE(nn.Module):
    def __init__(self,
        input_len: int,
        layers: int,
        risk_num: int,
        mlp_level:int,
        task_support_expert_num: int,
        layer_out_len: List[int],
        risk_task_expert_num: int,
        activation:str = "RELU",
    ) -> None:
        super().__init__()
        self.input_dim = input_len
        self.layers = layers
        self.risk_num = risk_num
        self.task_support_expert_num = task_support_expert_num
        self.risk_task_expert_num = risk_task_expert_num
        per_risk_task_expert_num = [risk_task_expert_num]*risk_num
        per_risk_task_expert_num.append(task_support_expert_num)
        self.per_risk_task_expert_num = per_risk_task_expert_num
        self.layer_experts = torch.nn.ModuleList().to(device)
        self.layer_inte_weight = torch.nn.ModuleList().to(device)
        self.layer_gate = torch.nn.ModuleList().to(device)
        for layer in range(self.layers):
            per_layer_expert = torch.nn.ModuleList().to(device)
            per_layer_gate = torch.nn.ModuleList().to(device)
            per_layer_inte_weight = []
            for expert_num in per_risk_task_expert_num:
                per_layer_expert.append(nn.ModuleList(
                    [MLP(input_len,mlp_level,layer_out_len[layer], activation)
                ]*expert_num))
                init_empty_weight = torch.empty(1, expert_num)
                if expert_num <= 0:
                    raise ValueError("expert_num 必须是正整数，但当前值为: {}".format(expert_num))

                uniform_scale = sqrt(1.0 / expert_num)
                torch.nn.init.uniform_(init_empty_weight, a=-uniform_scale, b=uniform_scale)
                param = torch.nn.Parameter( init_empty_weight)
                per_layer_inte_weight.append(param)
            num_gate = (len(per_risk_task_expert_num)-1) if layer == (len(per_risk_task_expert_num)-1) else len(per_risk_task_expert_num)
            per_layer_gate.extend([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        input_len, len(per_risk_task_expert_num)
                    ).to(device),
                    torch.nn.Softmax(dim=-1).to(device),                                                      
                ).to(device)
            ]*num_gate)
            self.layer_experts.append(per_layer_expert)   
            self.layer_gate.append(per_layer_gate)   
            self_exp_weights = nn.ParameterList(per_layer_inte_weight).to(device)
            self.layer_inte_weight.append(self_exp_weights)
            
    def forward(self,
        inputs: torch.Tensor,
    )-> torch.Tensor:
        for layer_idx in range(self.layers):
            cur_layer_task_expert = self.layer_experts[layer_idx]
            cur_layer_inte_weight = self.layer_inte_weight[layer_idx]
            cur_layer_gate = self.layer_gate[layer_idx]
            task_expert_out = []
            for i in range(len(self.per_risk_task_expert_num)):
                cur_task_experts_out = torch.stack(
                    [
                        expert(inputs[:, i, :]).to(device)
                        for idx,expert in enumerate(cur_layer_task_expert[i])
                    ],
                    dim=1,
                ) 
                result_out = torch.einsum('bex,e->bx', cur_task_experts_out,torch.squeeze(cur_layer_inte_weight[i],dim=0))
                task_expert_out.append(result_out)
            task_expert_out = torch.stack(task_expert_out,dim=1) 
            gates = torch.stack(
                [
                    gate_weight(task_expert_out[:, idx, :])
                    for idx, gate_weight in enumerate(cur_layer_gate)
                ],
                dim=1,
            )
            gate_out = torch.bmm(
                gates,
                task_expert_out,
            ) 

            inputs = gate_out+task_expert_out

        return inputs
   
class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        mlp_level:int,
        out_dim: int,
        activation: str = "RELU",
        bias: bool = True,
    ) -> None:
        super().__init__()

        mlp_net = []
        for mlp_dim in range(mlp_level):
            mlp_net.append(
                nn.Linear(in_features=input_dim, out_features=out_dim, bias=bias).to(device)
            )
            mlp_net.append(nn.ReLU().to(device))
            input_dim = out_dim
        self.mlp_net = nn.Sequential(*mlp_net)
        self.mlp_net = self.mlp_net.to(device)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        torch.cuda.set_device(0)
        return self.mlp_net(input.to(device)).to(device)
