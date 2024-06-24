import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

class Aggregator(nn.Module):

    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, device: str = 'cpu'):
        """
        Parameters
        ----------
        input_dim (Optional[int]) Dimension of input node features. Used for defining FC layer in pooling aggregators.
        output_dim : int or None
        """
        super(Aggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows]

        n = _len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        raise NotImplementedError

class MeanAggregator(Aggregator):

    def _aggregate(self, features):
        return torch.mean(features, dim=0)

class PoolAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        super(PoolAggregator, self).__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise NotImplementedError

class MaxPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        return torch.max(features, dim=0)[0]

class MeanPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        return torch.mean(features, dim=0)

class LSTMAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features: torch.Tensor):
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out