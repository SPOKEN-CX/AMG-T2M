"""HumanML3D caption lengths with frozen word features and dependency graphs."""
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data


class CaptionLengthDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.rows = [json.loads(s) for s in (self.root / f'{split}.jsonl').read_text().splitlines()]
        self.graphs = json.loads((self.root / 'graphs.json').read_text())
        self.features = np.load(self.root / 'features.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        graph = self.graphs[row['feature_index']]
        offset, nodes = graph['offset'], graph['nodes']
        edges = torch.tensor(graph['edges'], dtype=torch.long).reshape(-1, 2).t().contiguous()
        return Data(x=torch.from_numpy(self.features[offset:offset+nodes].astype(np.float32)),
                    edge_index=edges, word_count=torch.tensor([graph['word_count']], dtype=torch.float32),
                    y=torch.tensor([row['length_frames']], dtype=torch.float32),
                    # PyG offsets attributes containing 'index' by node counts.
                    # A dataset record ID must remain unchanged when batching.
                    record_id=torch.tensor([index], dtype=torch.long))
