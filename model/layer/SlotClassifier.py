import torch.nn as nn
from .biaffine import BiaffineLayer
from .feedForward import FeedforwardLayer


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slot_labels, dropout=0.0):
        super(SlotClassifier, self).__init__()
        self.feedStart = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.feedEnd = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.biaffine = BiaffineLayer(inSize1=input_dim, inSize2=input_dim, classSize=num_slot_labels)

    def forward(self, x):
        start = self.feedStart(x)
        end = self.feedEnd(x)
        score = self.biaffine(start, end)
        return score
