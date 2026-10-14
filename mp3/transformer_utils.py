import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Compute self attention by scaled dot product.
    ``query``, ``key``, and ``value`` are computed from input token features
    using linear layers. Similarity is computed using Scaled Dot-Product
    Attention where the dot product is scaled by a factor of square root of the
    dimension of the query vectors. See ``Attention Is All You Need" for more details.

    Args for __init__:
        input_dim (int): input dimention of attention
        query_dim (int): query dimention of attention
        key_dim (int): key dimention of attention
        value_dim (int): value dimention of attention

    Inputs for forward function:
        x (batch, num_tokens, input_dim): batch of input feature vectors for the tokens.
    Outputs from forward function:
        attn_output (batch, num_tokens, value_dim): outputs after self-attention
    """

    def __init__(self, input_dim, query_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()
        assert query_dim == key_dim
        self.query_dim = query_dim
        self.input_dim = input_dim

        self.W_query = nn.Linear(input_dim, query_dim)
        self.W_key = nn.Linear(input_dim, key_dim)
        self.W_value = nn.Linear(input_dim, value_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        print(x.size())

        # print(x)
        # (batch_size, seq_len, input_dim)

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        dk = K.size(2)
        # print(dk)
        mult = torch.matmul(Q, K.mT)
        # print(mult.size())
        dk_sq = dk ** (0.5)
        attention = self.softmax(mult / dk_sq) @ V

        return attention


class LayerNorm(nn.Module):
    """
    Args: input_dim, epsilon
        input_dim (int): dimensionality of input feature vectors
        epsilon (float): epsilon for when normalizing by the variance.

    Input to forward function:
        x (batch, num_tokens, input_dim): input features for tokens.

    Output from forward function:
        x_out (batch, num_tokens, input_dim): token features after layer normalization.
    """

    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        assert isinstance(input_dim, int)

        self.input_dim = input_dim
        self.eps = eps

        # w: the learnable weights initialized to 1.
        self.w = nn.Parameter(torch.ones(self.input_dim))

        # b: the learnable bias initialized to 0.
        self.b = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.input_dim
        # print(x.shape)  # [32, 12, 16])
        batch_size = x.size(0)
        seq_len = x.size(1)
        C = x.size(2)
        # input: (batch_size: N, seq_length: C, hidden_dim: D)
        # 1. calculate the mean of all elements (make sure you're taking the mean and variation over the d_model dimension)
        expected_val_c = torch.mean(x, dim=2, keepdim=True)
        # print("expected_val_c.size()")
        # print(expected_val_c.size())
        # 2. calculate the variance of all elements. Use the biased variance
        #    (normalize by the number of features, i.e. divide by C not C-1;
        #    pass unbiased=False if you use torch.var). This matches PyTorch's
        #    nn.LayerNorm and is what the tests expect.
        var = torch.var(x, dim=2, unbiased=False, keepdim=True)
        # print("var.size()")
        # print(var.size())

        # 3. calculate normalized x
        eq = (x - expected_val_c) / torch.sqrt(var + self.eps)
        # print(eq.size())
        # print(self.w.size())
        # print(self.b.size())

        # 4. apply scale and shift(the w and b parameters)
        ln = self.w * eq + self.b

        return ln
