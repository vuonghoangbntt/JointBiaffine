import torch
import torch.nn as nn
from .biaffine import BiaffineLayer
from .feedForward import FeedforwardLayer
from .attention import  Attention

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slot_labels, num_intent_labels, use_attention=False, attention_type= "soft", dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.use_attention = use_attention
        self.attention_type = attention_type
        if use_attention:
            self.attention = Attention(input_dim)
            self.linear_intent_context = nn.Linear(num_intent_labels, input_dim)
        if self.attention_type == 'hard':
            self.linear_out = nn.Linear(input_dim*2, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.feedStart = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.feedEnd = FeedforwardLayer(d_in=input_dim, d_hid=hidden_dim)
        self.biaffine = BiaffineLayer(inSize1=input_dim, inSize2=input_dim, classSize=num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        if self.use_attention and self.attention_type=='soft':
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)  # 1: query length (each token)
            output, weights = self.attention(x, intent_context, attention_mask)
            x = self.dropout(output)
        elif self.use_attention and self.attention_type=='hard':
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            intent_context = intent_context.expand(-1, x.size()[1], -1)
            x = torch.cat((x, intent_context), dim=2)
            x = self.linear_out(x)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        score = self.biaffine(start, end)
        return score
