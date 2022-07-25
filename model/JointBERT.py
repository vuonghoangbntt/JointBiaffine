import torch.nn as nn
from model.layer import MergeEmbedding, IntentClassifier, SlotClassifierWithCRF
from transformers import AutoConfig
from torchcrf import CRF


class JointBERT(nn.Module):
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
        self.intent_classifier = IntentClassifier(input_dim=args.hidden_dim * args.lstm_layers,
                                                  num_intent_labels=args.num_intent_labels)
        self.slot_classifier = SlotClassifierWithCRF(input_dim=args.hidden_dim, num_slot_labels=args.num_slot_labels)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids=None, char_ids=None, first_subword=None, attention_mask=None):
        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                          first_subword=first_subword,
                          char_ids=char_ids)
        x, (hn, cn) = self.bilstm(x)
        intent_logits = self.intent_classifier(hn.permute(1, 0, 2).reshape(x.shape[0], -1))
        slot_logits = self.slot_classifier(x)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
