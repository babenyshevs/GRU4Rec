"""Core neural network components used by GRU4Rec."""

import math
import numpy as np
import torch
from torch import nn


def init_parameter_matrix(
    tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1
):
    """Initialize ``tensor`` with a scaled uniform distribution."""
    sigma = math.sqrt(
        6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale)
    )
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)


class GRUEmbedding(nn.Module):
    """Single GRU step implemented with learnable embeddings."""

    def __init__(self, dim_in: int, dim_out: int):
        """Create embedding matrices for the first GRU layer."""
        super().__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(
            torch.empty((dim_out, dim_out * 2), dtype=torch.float)
        )
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embedding weights."""
        init_parameter_matrix(self.Wx0.weight, dim1_scale=3)
        init_parameter_matrix(self.Wrz0, dim1_scale=2)
        init_parameter_matrix(self.Wh0, dim1_scale=1)
        nn.init.zeros_(self.Bh0)

    def forward(self, X, H):
        """Perform a GRU update using embeddings ``X`` and hidden state ``H``."""
        Vx = self.Wx0(X) + self.Bh0
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z)
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h


class GRU4RecModel(nn.Module):
    """Stack of GRU layers with optional embeddings for GRU4Rec."""

    def __init__(
        self,
        n_items,
        layers=[100],
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        embedding=0,
        constrained_embedding=True,
    ):
        """Configure network dimensions and embedding behaviour."""
        super().__init__()
        self.n_items = n_items
        self.layers = layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = nn.Embedding(n_items, embedding, sparse=True)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1
        self.DE = nn.Dropout(dropout_p_embed)
        self.G = []
        self.D = []
        for i in range(self.start, len(layers)):
            self.G.append(
                nn.GRUCell(layers[i - 1] if i > 0 else n_input, layers[i])
            )
            self.D.append(nn.Dropout(dropout_p_hidden))
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)
        self.Wy = nn.Embedding(n_items, layers[-1], sparse=True)
        self.By = nn.Embedding(n_items, 1, sparse=True)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Reset parameters with uniform initialization."""
        if self.embedding:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale=3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale=3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)

    def _init_numpy_weights(self, shape):
        """Return a numpy array with Xavier-style random weights."""
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype("float32") * 2 * sigma - sigma
        return m

    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        """Reproduce the original paper's random initialisation scheme."""
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(
                torch.tensor(
                    self._init_numpy_weights((self.n_items, n_input)),
                    device=self.E.weight.device,
                )
            )
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(
                torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(
                torch.tensor(
                    self._init_numpy_weights((self.layers[0], self.layers[0])),
                    device=self.GE.Wh0.device,
                )
            )
            self.GE.Bh0.set_(
                torch.zeros((self.layers[0] * 3,), device=self.GE.Bh0.device)
            )
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(
                torch.tensor(np.vstack(m), device=self.G[i].weight_ih.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            self.G[i].weight_hh.set_(
                torch.tensor(np.vstack(m2), device=self.G[i].weight_hh.device)
            )
            self.G[i].bias_hh.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_hh.device)
            )
            self.G[i].bias_ih.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_ih.device)
            )
        self.Wy.weight.set_(
            torch.tensor(
                self._init_numpy_weights((self.n_items, self.layers[-1])),
                device=self.Wy.weight.device,
            )
        )
        self.By.weight.set_(
            torch.zeros((self.n_items, 1), device=self.By.weight.device)
        )

    def embed_constrained(self, X, Y=None):
        """Use output embeddings for both input and output representations."""
        if Y is not None:
            XY = torch.cat([X, Y])
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_separate(self, X, Y=None):
        """Use a dedicated embedding matrix for item inputs."""
        E = self.E(X)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_gru(self, X, H, Y=None):
        """Embed items using a GRU cell when no separate embeddings exist."""
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed(self, X, H, Y=None):
        """Return embeddings for items and outputs based on configuration."""
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B

    def hidden_step(self, X, H, training=False):
        """Propagate input ``X`` through GRU layers updating ``H``."""
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, H[i])
            if training:
                X = self.D[i](X)
            H[i] = X
        return X

    def score_items(self, X, O, B):
        """Compute scores for each item embedding in ``O``."""
        O = torch.mm(X, O.T) + B.T
        return O

    def forward(self, X, H, Y, training=False):
        """Run a forward pass and return item scores."""
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)
        if not (self.constrained_embedding or self.embedding):
            H[0] = E
        Xh = self.hidden_step(E, H, training=training)
        R = self.score_items(Xh, O, B)
        return R
