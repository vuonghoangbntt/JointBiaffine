import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    """Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(32, 50, 256)
         >>> context = torch.randn(32, 1, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([32, 50, 256])
         >>> weights.size()
         torch.Size([32, 50, 1])
    """

    def __init__(self, dimensions):
        super(Attention, self).__init__()

        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
            output length: length of utterance
            query length: length of each token (1)
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        # query = self.linear_query(query)

        batch_size, output_len, hidden_size = query.size()
        # query_len = context.size(1)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        if attention_mask is not None:
            # Create attention mask, apply attention mask before softmax
            attention_mask = torch.unsqueeze(attention_mask, 2)
            # attention_mask = attention_mask.view(batch_size * output_len, query_len)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        # attention_scores = torch.squeeze(attention_scores,1)
        attention_weights = self.softmax(attention_scores)
        # attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # from IPython import embed; embed()
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        # combined = combined.view(batch_size * output_len, 2 * self.dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        # output = self.linear_out(combined).view(batch_size, output_len, self.dimensions)
        output = self.linear_out(combined)

        output = self.tanh(output)
        # output = combined
        return output, attention_weights
