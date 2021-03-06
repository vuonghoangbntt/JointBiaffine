import torch.nn as nn
from model.layer import MergeEmbedding, IntentClassifier, SlotClassifier
from transformers import AutoConfig


class JointBiaffine(nn.Module):
    def __init__(self, args):
        super(JointBiaffine, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.args = args
        self.num_slot_labels = args.num_slot_labels
        self.num_intent_labels = args.num_intent_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        self.word_rep = MergeEmbedding(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=args.lstm_layers, bidirectional=True, batch_first=True)
        self.intent_classifier = IntentClassifier(input_dim=args.hidden_dim * args.lstm_layers,
                                                  num_intent_labels=args.num_intent_labels)
        self.slot_classifier = SlotClassifier(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim_ffw,
                                              num_slot_labels=args.num_slot_labels, num_intent_labels=args.num_intent_labels,
                                              use_attention=args.use_attention, attention_type=args.attention_type)

    def forward(self, input_ids=None, char_ids=None, first_subword=None, attention_mask=None):
        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                          first_subword=first_subword,
                          char_ids=char_ids)
        x, (hn, cn) = self.bilstm(x)
        intent_output = self.intent_classifier(hn.permute(1, 0, 2).reshape(x.shape[0], -1))
        if not self.args.use_attention and self.args.attention_type == 'hard':
            slot_output = self.slot_classifier(x, hn.permute(1, 0, 2).reshape(x.shape[0], -1), attention_mask)
        else:
            slot_output = self.slot_classifier(x, intent_output, attention_mask)
        return intent_output, slot_output
