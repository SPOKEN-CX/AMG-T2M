import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
import clip
import spacy

# --------------------------------------------------------
# 工具函数
# --------------------------------------------------------
def init_weight(layer):
    if isinstance(layer, (nn.Linear, nn.Embedding)):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# --------------------------------------------------------
# 模型内部组件
# --------------------------------------------------------
class StylizationBlock(nn.Module):
    """
    在 x 上做 LayerNorm + (1+scale)*x + shift，然后再过一层 MLP。
    其中 scale, shift 从 emb 拆分而来。
    """
    def __init__(self, latent_dim, step_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(step_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, h, emb):
        """
        h: [N, latent_dim]
        emb: [N, latent_dim]  (已经对batch做广播)
        """
        emb = emb.to(self.emb_layers[1].weight.device)
        emb_out = self.emb_layers(emb)  # [N, 2*latent_dim]
        scale, shift = torch.chunk(emb_out, 2, dim=-1)  # 都是 [N, latent_dim]

        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class GraphLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        # x: [C, N] 或 [N, C] 要注意 transpose
        # 但我们在 GraphResBlock 里会处理 transpose
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphResBlock(nn.Module):
    """
    使用多头GAT做 Residual。输入/输出都是 x: [N, d].
    """
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv1 = GATConv(out_channels // 2, out_channels // 2, heads=self.num_heads, concat=False)
        self.conv2 = GATConv(out_channels // 2, out_channels // 2, heads=self.num_heads, concat=False)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.LayerNorm(in_channels)
        self.norm1 = nn.LayerNorm(out_channels // 2)
        self.norm2 = nn.LayerNorm(out_channels // 2)

    def forward(self, data):
        """
        data: PyG Batch 对象, 其 data.x: [N, in_channels], data.edge_index: [2, E]
        最终更新 data.x = [N, out_channels]
        """
        x, edge_index = data.x, data.edge_index
        device = self.pre_norm.weight.device
        x = x.to(device)
        edge_index = edge_index.to(device)

        # 先 LN + ReLU, 再过 GraphLinear -> [N, out_channels // 2]
        y = F.relu(self.pre_norm(x))
        # lin1 的实现里 x 的形状期望有注意，所以我们转一下
        # GraphLinear 里 forward 写得比较怪, 这里统一做法:
        y = y.transpose(0, 1)  # [d, N]
        y = self.lin1(y)       # -> [out_channels // 2, N]
        y = y.squeeze(0).transpose(0, 1)  # -> [N, out_channels // 2]

        # 两次多头GAT
        y = self.conv1(y, edge_index)
        y = self.conv2(y, edge_index)

        # LN + ReLU
        y = F.relu(self.norm2(y))
        # 再过 lin2
        y = y.transpose(0, 1)   # -> [d2, N]
        y = self.lin2(y)       # -> [out_channels, N]
        y = y.transpose(1, 2).squeeze(0)  # -> [N, out_channels]

        # residual
        z = x + y
        data.x = z
        return data.x

class NormalFFN(nn.Module):
    """
    你原先的 FFN。期望输入 [B, T, latent_dim]，逐 token 做非线性，然后对 T 做平均得到 [B, output_dim]。
    """
    def __init__(self, latent_dim, output_dim, dropout):
        super().__init__()
        self.ResLinear = nn.Linear(latent_dim, output_dim)
        self.output = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, T, latent_dim]
        1) 过 ResLinear -> [B, T, output_dim]
        2) 过 self.output -> [B, T, output_dim]
        3) (x1 + y) 做均值 -> [B, output_dim]
        4) Sigmoid
        """
        x1 = self.ResLinear(x)  # shape: (B, T, output_dim)
        y = self.output(x)      # shape: (B, T, output_dim)
        result = torch.mean(x1 + y, dim=1)  # shape: (B, output_dim)
        return torch.sigmoid(result)


class MotionLenEstimatorGraphTransformer(nn.Module):
    def __init__(self,
                 clip_dim=512,
                 latent_dim=512,
                 output_dim=1,
                 step_emb_dim=512,
                 num_Res_layers=4,
                 clip_version='ViT-B/32',
                 num_heads=4,
                 dropout=0.3):
        super(MotionLenEstimatorGraphTransformer, self).__init__()
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 载入 spacy
        model_path = "/path/to/spacy/en_core_web_sm"
        self.nlp = spacy.load(model_path)

        # 载入 CLIP, 并冻结
        self.clip, _ = clip.load(clip_version, self.device)
        self.clip.to(self.device)
        set_requires_grad(self.clip, False)

        self.num_Res_layers = num_Res_layers

        # 构造多头GAT的残差模块
        self.graph_res_blocks = nn.ModuleList(
            [GraphResBlock(clip_dim, latent_dim, num_heads) for _ in range(self.num_Res_layers)]
        )

        # 对每层结果都做 Stylization
        self.stylization_blocks = nn.ModuleList(
            [StylizationBlock(latent_dim, latent_dim, dropout) for _ in range(num_Res_layers)]
        )

        # 最后的回归/分类FFN
        self.normffn = NormalFFN(latent_dim, output_dim, dropout)

        # 对 caption length 做 embedding
        self.cap_len_emb = nn.Linear(1, latent_dim)
        self.cap_len_emb.apply(init_weight)

        self.output_dim = output_dim

    def encode_text(self, raw_text_list, max_len=77):
        """
        手动获取 CLIP 的 token-level hidden states,
        得到 shape [batch_size, max_len, 512]。
        """
        text = clip.tokenize(raw_text_list, truncate=True).to(self.device)
        with torch.no_grad():
            # 1) token embedding + positional embedding
            feat_clip_text = self.clip.token_embedding(text).type(self.clip.dtype)
            feat_clip_text = feat_clip_text + self.clip.positional_embedding.type(self.clip.dtype)
            # 2) transformer 多层
            feat_clip_text = feat_clip_text.permute(1, 0, 2)  # -> [sequence_length, batch_size, 512]
            feat_clip_text = self.clip.transformer(feat_clip_text)  # 多层自注意力
            feat_clip_text = feat_clip_text.permute(1, 0, 2)  # -> [batch_size, sequence_length, 512]
            # 3) 截断到 max_len
            feat_clip_text = feat_clip_text[:, :max_len, :]
        feat_clip_text = feat_clip_text.float()
        return feat_clip_text  # [B, max_len, 512]

    def get_dependency_edges(self, sentences, max_len=77):
        """
        用 spacy 做依存分析，每个句子返回一个边列表: [(child, head), ...]
        """
        dependency_edges = []
        token_counts = []
        # batch 解析
        docs = self.nlp.pipe(sentences, batch_size=len(sentences))
        for doc in docs:
            if len(doc) > max_len:
                doc = doc[:max_len]
            edges = [(token.i, token.head.i) for token in doc if token.head.i != token.i]
            dependency_edges.append(edges)
            token_counts.append(len(doc))
        return dependency_edges, token_counts

    # --------------------------------------
    # 构建 PyG Batch
    # --------------------------------------
    def build_semantic_graph(self, sentences, device, max_len=77):
        """
        返回一个 Batch 对象，包含整个 batch 的图数据
        """
        # 1) 获取 token-level 特征 [B, max_len, 512]
        feat_clip_text = self.encode_text(sentences, max_len)
        # 2) 获取依存关系
        dependency_edges, token_counts = self.get_dependency_edges(sentences, max_len)

        data_list = []
        batch_size = len(sentences)
        for i in range(batch_size):
            n_tokens = token_counts[i]
            # 取前 n_tokens 的特征
            x_i = feat_clip_text[i, :n_tokens, :]  # [n_tokens, 512]

            # 构建 edge_index
            edges = dependency_edges[i]  # list of (child, head)
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                # 如果句子只有一个词 或者空，没法构建依存
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # 封装到 Data 对象
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)

        # 3) 合并为一个 Batch
        batch_data = Batch.from_data_list(data_list).to(device)
        return batch_data

    def forward(self, caption_list, cap_len):
        """
        caption_list: list of str, size B
        cap_len: [B], 每条 caption 的长度信息(或你要用的别的信息)
        """
        device = self.device

        # 1) 先把cap_len变成 latent_dim, shape [B, latent_dim]
        cap_len = cap_len.view(-1, 1).float().to(device)
        cap_len_emb = self.cap_len_emb(cap_len)  # [B, latent_dim]

        # 2) 一次性构建 batch 图
        batch_data = self.build_semantic_graph(caption_list, device)

        # 现在 batch_data.x: [N, 512] -> GNN -> [N, latent_dim]
        # 多层 GraphResBlock + StylizationBlock
        for j in range(self.num_Res_layers):
            # 先做 GraphResBlock
            batch_data.x = self.graph_res_blocks[j](batch_data)  # [N, latent_dim]
            # 给每个节点的 stylization emb
            # batch_data.batch: [N], 值在 [0, B-1]
            emb_expanded = cap_len_emb[batch_data.batch]  # [N, latent_dim]
            batch_data.x = self.stylization_blocks[j](batch_data.x, emb_expanded)

        # 此时 batch_data.x: [N, latent_dim]
        # 如果 NormalFFN 需要 [B, T, latent_dim] 形状，我们就把每个句子的 token 特征取出来并pad
        # 取出 batch_size
        B = len(caption_list)
        # 先统计一下各句子节点数
        node_counts = torch.bincount(batch_data.batch)  # [B], each entry is n_tokens_of_that_sentence
        max_len = torch.max(node_counts).item() if len(node_counts) > 0 else 1

        # 把 batch_data.x 还原到 [B, max_len, latent_dim] (padding)
        splitted_x = []
        start_idx = 0
        for i in range(B):
            length_i = node_counts[i]
            x_i = batch_data.x[start_idx : start_idx + length_i]  # [length_i, latent_dim]
            start_idx += length_i
            # pad到 max_len
            if length_i < max_len:
                pad_size = int(max_len - length_i)
                pad_zeros = torch.zeros((pad_size, self.latent_dim), device=device)
                x_i = torch.cat([x_i, pad_zeros], dim=0)
            splitted_x.append(x_i.unsqueeze(0))  # -> [1, max_len, latent_dim]

        # 合并
        splitted_x = torch.cat(splitted_x, dim=0)  # -> [B, max_len, latent_dim]

        # 3) 用 NormalFFN 输出 [B, output_dim]
        out = self.normffn(splitted_x)  # -> [B, 1]
        return out.squeeze(-1)



if __name__ == '__main__':
    clip_version = 'ViT-B/32'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Graphmodel = MotionLenEstimatorGraphTransformer(
        clip_dim=512,
        latent_dim=512,
        step_emb_dim=512,
        output_dim=1,
        num_Res_layers=4,
        clip_version=clip_version,
        num_heads=4,
        dropout=0.3
    ).to(device)

    caption = [
        'A person leaps over',
        'Someone is walking forward',
        'Someone is turning',
        'A person leaps over',
        'Someone is walking forward',
        'Someone is turning',
        'A person leaps over',
        'Someone is walking forward',
        'Someone is turning',
        'A person leaps over',
        'Someone is walking forward',
        'Someone is turning'
    ]

    # 假设每条句子的cap_len是随便给的
    cap_lens = torch.tensor([4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3])

    start_t = time.time()
    with torch.no_grad():
        # 前向
        graphs_t = Graphmodel(caption, cap_lens)
    end_t = time.time()

    print(f"Processed Graphs Shape: {graphs_t} , time taken: {end_t - start_t:.4f}s")
    # 预期: Processed Graphs Shape: torch.Size([12])  (因为output_dim=1, squeeze后是[12])
