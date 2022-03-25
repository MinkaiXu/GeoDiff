import torch
from torch.nn import Module
from torch_scatter import scatter_add, scatter_mean, scatter_max

from ..common import coarse_grain, batch_to_natoms, get_complete_graph
from .schnet import SchNetEncoder, GaussianSmearing


class CoarseGrainingEncoder(Module):

    def __init__(self, hidden_channels, num_filters, num_interactions, edge_channels, cutoff, smooth):
        super().__init__()
        self.encoder = SchNetEncoder(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            edge_channels=edge_channels,
            cutoff=cutoff,
            smooth=smooth,
        )
        self.distexp = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)


    def forward(self, pos, node_attr, subgraph_index, batch, return_coarse=False):
        """
        Args:
            pos:    (N, 3)
            node_attr:  (N, H)
            subgraph_index:  (N, )
            batch:  (N, )
        """
        cluster_pos, cluster_attr, cluster_batch = coarse_grain(pos, node_attr, subgraph_index, batch)

        edge_index, _ = get_complete_graph(batch_to_natoms(cluster_batch))
        row, col = edge_index
        edge_length = torch.norm(cluster_pos[row] - cluster_pos[col], dim=1, p=2)
        edge_attr = self.distexp(edge_length)

        h = self.encoder(
            z = cluster_attr,
            edge_index = edge_index,
            edge_length = edge_length,
            edge_attr = edge_attr,
            embed_node = False,
        )

        if return_graph:
            return h, cluster_pos, cluster_attr, cluster_batch, edge_index
        else:
            return h
