from transformers import AutoTokenizer
from dataloader import MyDataSet
from trainer import Trainer
import argparse
import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args):
    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = MyDataSet(path=args.train_path, char_vocab_path=args.char_vocab_path,
                              tokenizer=tokenizer, intent_label_path=args.intent_label_path,
                              slot_label_path=args.slot_label_path, max_char_len=args.max_char_len,
                              max_seq_length=args.max_seq_length)
    dev_dataset = MyDataSet(path=args.dev_path, char_vocab_path=args.char_vocab_path,
                            tokenizer=tokenizer, intent_label_path=args.intent_label_path,
                            slot_label_path=args.slot_label_path, max_char_len=args.max_char_len,
                            max_seq_length=args.max_seq_length)

    test_dataset = MyDataSet(path=args.test_path, char_vocab_path=args.char_vocab_path,
                             tokenizer=tokenizer, intent_label_path=args.intent_label_path,
                             slot_label_path=args.slot_label_path, max_char_len=args.max_char_len,
                             max_seq_length=args.max_seq_length)

    trainer = Trainer(args=args, train_dataset=train_dataset,
                      dev_dataset=dev_dataset, test_dataset=test_dataset)
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.logger.info('Dev Result: ')
        trainer.eval("dev")
        trainer.logger.info('Test Result:')
        trainer.eval("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_path", default="data/SF_ATIS/train.json", type=str)
    parser.add_argument('--dev_path', type=str, default="data/SF_ATIS/dev.json")
    parser.add_argument('--test_path', type=str, default="data/SF_ATIS/test.json")
    parser.add_argument('--char_vocab_path', type=str, default="data/charindex.json")
    parser.add_argument('--intent_label_path', type=str, default="data/SF_ATIS/intent_label.txt")
    parser.add_argument('--slot_label_path', type=str, default="data/SF_ATIS/slot_label.txt")
    parser.add_argument('--max_char_len', default=20, type=int)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # model
    parser.add_argument('--use_char', action="store_true")
    parser.add_argument('--use_attention', action="store_true")
    parser.add_argument('--char_embedding_dim', default=100, type=int)
    parser.add_argument('--char_hidden_dim', default=200, type=int)
    parser.add_argument('--num_layer_bert', default=1, type=int)
    parser.add_argument('--char_vocab_size', default=108, type=int)
    parser.add_argument('--hidden_dim', default=728, type=int)
    parser.add_argument('--hidden_dim_ffw', default=400, type=int)
    parser.add_argument('--num_intent_labels', default=25, type=int)
    parser.add_argument('--num_slot_labels', default=82, type=int)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--intent_weight', type=float, default=0.2)
    parser.add_argument('--attention_input_dim', type=int, default=300)

    # train
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument("--use_dice_loss", action="store_true")
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")

    parser.add_argument('--save_folder', default='results', type=str)
    parser.add_argument('--seed', default=1, type=int)
    args, unk = parser.parse_known_args()

    train(args)
