import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import spacy
import numpy as np
import clip

class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

def get_dependency_edges(nlp, sentences):
    dependency_edges = []
    token_counts = []
    for sentence in sentences:
        doc = nlp(sentence)
        edges = [(token.i, token.head.i) for token in doc if token.head.i != token.i]
        dependency_edges.append(edges)
        token_counts.append(len(doc))
    return dependency_edges, token_counts

def build_semantic_graph(nlp, sentences, clip_moel, device):
    dependency_edges, token_counts = get_dependency_edges(nlp, sentences)
    graphs = []

    text = clip.tokenize(sentences, truncate=True).to(device)
    feat_clip_text = clip_model.token_embedding(text).type(torch.float32).to(device)
    print('feat_clip_text', feat_clip_text.shape)

    for i, sentence in enumerate(sentences):
        doc = nlp(sentence)
        n_tokens = token_counts[i]

        # 获取前 n_tokens 个词的 CLIP 嵌入
        x = feat_clip_text[i, :n_tokens, :]  # [n_tokens, 512]

        edge_index = torch.tensor(dependency_edges[i], dtype=torch.long).t().contiguous().to(device)

        data = Data(x=x, edge_index=edge_index)
        graphs.append(data)

    return graphs

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_version = 'ViT-B/32'
    clip_model, _ = clip.load(clip_version, device=device)
    raw_text = ["A man lifts an object with both hands", "A person is walking"]

    model_path = "../en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0"
    nlp = spacy.load(model_path)

    graphs = build_semantic_graph(nlp, raw_text, clip_model, device)

    for i, graph in enumerate(graphs):
        print(f"Graph {i}:")
        print("Number of nodes:", graph.num_nodes)
        print("Number of edges:", graph.num_edges)
        print("Node features shape:", graph.x.shape)
        print("Edge index shape:", graph.edge_index.shape)

    # 假设输入的特征维度是 CLIP 的输出维度
    in_channels = graphs[0].x.size(1)
    out_channels = in_channels  # 你可以根据具体任务设置输出特征的维度

    model = GNNModel(in_channels, out_channels).to(device)

    for i, graph in enumerate(graphs):
        x, edge_index = graph.x.to(device), graph.edge_index.to(device)
        print(f"Original features of graph {i}:")
        print(x)

        out = model(x, edge_index)
        print(f"Output features of graph {i}:")
        print(out.shape)

        # Check if the edge index has changed
        print(f"Edge index of graph {i} (should be unchanged):")
        print(edge_index)
