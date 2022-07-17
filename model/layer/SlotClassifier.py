import torch
import torch.nn as nn
from .biaffine import BiaffineLayer
from .feedForward import FeedforwardLayer
from .attention import  Attention

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slot_labels, use_attention=False, attention_input_dim= None, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        if use_attention:
            self.attention = Attention(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.feedStart = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.feedEnd = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.biaffine = BiaffineLayer(inSize1=input_dim, inSize2=input_dim, classSize=num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        intent_context = torch.unsqueeze(intent_context, 1)  # 1: query length (each token)
        output, weights = self.attention(x, intent_context, attention_mask)
        x = self.dropout(output)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        score = self.biaffine(start, end)
        return score
