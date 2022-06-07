import os
import json


def format_PhoATIS_file(data_path, output_path, input_file="seq.in", output_file="seq.out"):
    sentences = []
    labels = []
    json_format = []
    with open(os.path.join(data_path, input_file), encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            sentences.append(line.strip())
    with open(os.path.join(data_path, output_file), encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            label = []
            start = -1
            end = 0
            current_label = "O"
            for idx, token in enumerate(line.split()):
                if token[0] == "B":
                    end = idx - 1
                    if start > 0:
                        label.append([current_label, start, end])
                    start = idx
                    current_label = token[2:]
                elif token[0] == "O":
                    end = idx - 1
                    if start > 0:
                        label.append([current_label, start, end])
                    start = -1
                    current_label = "O"
            if start != -1:
                label.append([current_label, start, idx])
            labels.append(label)
    intent_labels = []
    with open(os.path.join(data_path, "label"), encoding="utf-8") as f:
        intent_labels = [line.strip() for line in f.readlines()]

    assert len(intent_labels) == len(sentences)
    assert len(labels) == len(sentences)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sent, label, intent in zip(sentences, labels, intent_labels):
            json.dump({"sentence": sent, "slot label": label, "intent label": intent}, outfile, ensure_ascii=False)
            outfile.write('\n')


def get_label_set(data_path, output_path):
    label_list = []
    with open(data_path) as f:
        for line in f.readlines():
            if line.startswith("B") or line.startswith("I"):
                label_list.append(line.strip()[2:])
            else:
                label_list.append(line.strip())
    label_list = set(label_list)
    with open(output_path, 'w') as f:
        f.write('\n'.join(label_list))


#format_PhoATIS_file("./data/snips/test", output_path="./data/snips/test.json")
#format_PhoATIS_file("./data/snips/train", output_path="./data/snips/train.json")
#format_PhoATIS_file("./data/snips/valid", output_path="./data/snips/dev.json")

get_label_set('./data/atis/slot_label.txt', './data/atis/slot_label.txt')
#get_label_set('./data/snips/slot_label.txt', './data/snips/slot_label.txt')
