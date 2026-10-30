import torch.nn as nn
import torch
import torch.nn.functional as F


class StepWeightedTransformer(nn.Module):
    def __init__(self, dim_motion, trans_num, num_vq, num_heads=4):
        super(StepWeightedTransformer, self).__init__()

        self.dim_motion = dim_motion
        self.hidden_dim = 256
        self.num_heads = num_heads
        self.num_vq = num_vq

        self.input_proj = nn.Linear(dim_motion, self.hidden_dim)
        self.transformer = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads)
        self.output_proj = nn.Linear(self.hidden_dim, dim_motion)

        # 增加一个线性层用于生成码本向量的概率分布
        self.vq_classifier = nn.Linear(dim_motion, self.num_vq)

    def forward(self, seq_before):
        batch_size, seq_len, _ = seq_before.size()
        if seq_len == 0:
            raise ValueError('At least one history token is required')

        # 输入映射
        x = self.input_proj(seq_before)  # (batch, seq_len, hidden_dim)
        x = x.transpose(0, 1)  # 转换为(seq_len, batch, hidden_dim)

        attn_output, _ = self.transformer(x, x, x)

        # 生成阶梯型距离权重（越靠近序列末尾的权重越大）
        distance_weights = (torch.ones(1, device=seq_before.device, dtype=seq_before.dtype)
                            if seq_len == 1 else
                            torch.linspace(0, 1, steps=seq_len, device=seq_before.device, dtype=seq_before.dtype))
        distance_weights = distance_weights.view(seq_len, 1, 1)

        # 应用距离权重
        weighted_output = attn_output * distance_weights

        # 聚合
        weighted_sum = weighted_output.sum(dim=0) / distance_weights.sum()

        # 生成过渡帧特征
        transition_feature = self.output_proj(weighted_sum)  # (batch, dim_motion)

        # 计算下一个码本向量的概率分布
        logits = self.vq_classifier(transition_feature)  # (batch, num_vq)
        return logits
