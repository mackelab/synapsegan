from typing import List, Optional
from warnings import warn
import torch
import numpy as np
from torch import nn


class Discriminator(nn.Module):
    """Base class for GAN discriminator."""

    def __init__(self,
                 hidden_layers: List[nn.Module],
                 conditional: Optional[bool] = True,
                 embed: Optional[bool] = False,
                 embed_input: Optional[List[nn.Module]] = None,
                 embed_output: Optional[List[nn.Module]] = None
                 ):
        """
        Set up discriminator.

        Args:
            hidden_layers: list of nn.Module objects that are fed into
                           nn.Sequential to build a discriminator network.
            conditional: if True, inputs are concatenated along the last
                         dimension before being passed through the hidden
                         layers.
            embed: if True, embed inputs befor passing them through hidden
                   layers.
            embed_input: list of nn.Module, for input embedding.
            embed_output: list of nn.Module, for output embedding.
        """
        super(Discriminator, self).__init__()
        self._hidden_layers = nn.Sequential(*hidden_layers)
        self.conditional = conditional
        self.embed = embed
        self.embed_input = embed_input
        self.embed_output = embed_output

        # self._check_last_layer()
        self._check_embed()
        self._make_embed()
        self._check_conditional()

    def forward(self, inputs: List[torch.tensor]) -> torch.tensor:
        """
        Forward pass.

        Args:
            inputs: list of inputs; the first should be the input to the
                    generator, the second should be the output of the
                    generator.
        """
        input_to_hidden_layers = None
        if self.conditional:
            assert(isinstance(inputs, list))
            if self.embed:
                input_to_hidden_layers = self._get_embeddings(*inputs)
                input_to_hidden_layers = torch.cat(input_to_hidden_layers,
                                                   -1)
            else:
                input_to_hidden_layers = torch.cat(inputs, -1)
        else:
            input_to_hidden_layers = inputs
        return self._hidden_layers(input_to_hidden_layers)

    def _check_last_layer(self):
        if not isinstance(self._hidden_layers[-1], nn.Sigmoid):
            warn("Last layer is a Sigmoid")

    def _check_conditional(self):
        if self.embed:
            self.conditional = True

    def _check_embed(self):
        if self.embed and np.all([isinstance(net, type(None))
                                  for net in [self.embed_input,
                                              self.embed_output]
                                  ]
                                 ):
            warn("No embed networks were passed even though self.embed=True.")
            self.embed = False
        elif not self.embed and not np.all([isinstance(net, type(None))
                                            for net in [self.embed_input,
                                                        self.embed_output]
                                            ]
                                           ):
            self.embed = True

    def _make_embed(self):
        if not isinstance(self.embed_input, type(None)):
            self.embed_input = nn.Sequential(*self.embed_input)
        if not isinstance(self.embed_output, type(None)):
            self.embed_output = nn.Sequential(*self.embed_output)

    def _get_embeddings(self, _output, _input):
        embeddings = []
        for net, val in zip([self.embed_output, self.embed_input],
                            [_output, _input]):
            if isinstance(net, type(None)):
                embeddings.append(val)
            else:
                embeddings.append(net(val))
        return embeddings
