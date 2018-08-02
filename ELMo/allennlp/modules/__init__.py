"""
Custom PyTorch
`Module <http://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP
:class:`~allennlp.models.model.Model` s.
"""

from allennlp.modules.highway import Highway
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.time_distributed import TimeDistributed
# from allennlp.modules.token_embedders import TokenEmbedder, Embedding
