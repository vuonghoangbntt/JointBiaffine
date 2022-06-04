import itertools
from metrics import batch_computeF1
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import JointBiaffine
from tqdm import trange
from tqdm.notebook import tqdm
from dataloader import get_useful_ones, get_mask
import os
import logging


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.save_folder = os.path.join(args.save_folder, f'seed={args.seed}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        self.model = JointBiaffine(args=args)
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.best_score = 0
        self.intent_label_set = train_dataset.intent_label_set
        self.slot_label_set = train_dataset.slot_label_set
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler("{0}/{1}".format(self.save_folder, "log"),
                                           mode='w')
        self.logger.addHandler(file_handler)
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(consoleHandler)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=train_sampler,
                                      batch_size=self.args.batch_size, num_workers=16)

        total_steps = len(train_dataloader) * self.args.num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        for epoch in trange(self.args.num_epochs):
            train_loss = 0
            self.logger.info('\n------------------------------------')
            self.logger.info(f'\n          EPOCH: {epoch}           ')
            self.logger.info('\n------------------------------------')
            self.model.train()
            step = 0
            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'first_subword': batch[2],
                          'char_ids': batch[4],
                          }
                slot_label = batch[-1]
                intent_label = batch[-2]
                seq_length = batch[3]
                self.model.zero_grad()

                intent_logits, slot_score = self.model(**inputs)
                optimizer.zero_grad()
                mask = get_mask(max_length=self.args.max_seq_length, seq_length=seq_length)
                mask = mask.to(self.device)
                tmp_out, tmp_label = get_useful_ones(slot_score, slot_label, mask)

                loss = self.args.intent_weight * loss_func(intent_logits, intent_label) + (
                            1 - self.args.intent_weight) * loss_func(tmp_out, tmp_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # norm gradient
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=self.args.max_grad_norm)

                # update learning rate
                scheduler.step()
                step += 1
            self.logger.info(f'train loss:{train_loss / len(train_dataloader)}')
            self.eval('dev')

    def eval(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = RandomSampler(dataset)
        eval_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=self.args.batch_size,
                                     num_workers=16)

        self.model.eval()
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        eval_loss = 0
        intent_labels, slot_labels, slot_outputs, intent_outputs, seq_lengths = [], [], [], [], []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'first_subword': batch[2],
                      'char_ids': batch[4],
                      }
            slot_label = batch[-1]
            intent_label = batch[-2]
            seq_length = batch[3]
            self.model.zero_grad()

            with torch.no_grad():
                intent_logit, slot_score = self.model(**inputs)
            seq_lengths.append(seq_length)
            mask = get_mask(max_length=self.args.max_seq_length, seq_length=seq_length)
            mask = mask.to(self.device)

            tmp_out, tmp_label = get_useful_ones(slot_score, slot_label, mask)
            slot_labels.append(slot_label.cpu())
            intent_labels.append(intent_label.cpu())

            intent_outputs.append(torch.argmax(intent_logit, dim=1).cpu())
            slot_outputs.append(slot_score.cpu())

            loss = self.args.intent_weight * loss_func(intent_logit, intent_label) + (
                        1 - self.args.intent_weight) * loss_func(tmp_out, tmp_label)
            eval_loss += loss.item()
        slot_labels = torch.cat(slot_labels, dim=0)
        slot_outputs = torch.cat(slot_outputs, dim=0)
        seq_lengths = torch.cat(seq_lengths, dim=0)
        intent_labels = list(itertools.chain.from_iterable(intent_labels))
        intent_outputs = list(itertools.chain.from_iterable(intent_outputs))

        precision, recall, f1_score, report, intent_accuracy, frame_accuracy = batch_computeF1(intent_labels,
                                                                                               intent_outputs,
                                                                                               slot_labels,
                                                                                               slot_outputs,
                                                                                               seq_lengths,
                                                                                               self.slot_label_set,
                                                                                               self.save_folder,
                                                                                               do_error_analyze=(
                                                                                                           mode == 'test'),
                                                                                               samples=dataset.samples)

        result = {
            '{} loss'.format(mode): eval_loss / len(eval_dataloader),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'intent_accuracy': intent_accuracy,
            'frame_accuracy': frame_accuracy
        }
        if (f1_score + intent_accuracy) / 2 > self.best_score:
            self.save_model()
            self.best_score = (f1_score + intent_accuracy) / 2
        self.logger.info(str(result))
        # print(report)

    def save_model(self):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      }
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        torch.save(checkpoint, path)
        torch.save(self.args, os.path.join(self.args.save_folder, 'training_args.bin'))

    def load_model(self):
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])