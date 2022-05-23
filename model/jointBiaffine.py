import torch.nn as nn
from model.layer import MergeEmbedding, IntentClassifier, SlotClassifier
from transformers import AutoConfig


class JointBiaffine(nn.Module):
    def __init__(self, args):
        super(JointBiaffine, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_slot_labels = args.num_slot_labels
        self.num_intent_labels = args.num_intent_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        self.word_rep = MergeEmbedding(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=args.lstm_layers, bidirectional=True, batch_first=True)
        self.intent_classifier = IntentClassifier(input_dim=args.hidden_dim, num_intent_labels=args.num_intent_labels)
        self.slot_classifier = SlotClassifier(input_dim=args.hidden_dim * 2 * args.lstm_layers,
                                              hidden_dim=args.hidden_dim_ffw, num_slot_labels=args.num_slot_labels)

    def forward(self, input_ids=None, char_ids=None, first_subword=None, attention_mask=None):
        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                          first_subword=first_subword,
                          char_ids=char_ids)
        x, (hn, cn) = self.bilstm(x)
        return self.intent_classifier(hn.permute(1, 0, 2).reshape(x.shape[0], -1)), self.slot_classifier(x)
