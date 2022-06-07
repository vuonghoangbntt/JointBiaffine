import os
import json

with open("./data/snips/train/label",'r') as f:
    lines = f.readlines()
    labels = []
    for line in lines:
        if line.strip()!='':
            labels.append(line.strip())
    print('\n'.join(set(labels)))
with open("./data/snips/intent_label.txt",'w') as f:
    f.write('\n'.join(set(labels)))
with open('./data/snips/train/seq.out') as f:
    lines = f.readlines()
    labels = []
    for line in lines:
        if len(line.split())>0:
            labels.extend(line.split())
    print('\n'.join(set(labels)))
with open("./data/snips/slot_label.txt", 'w') as f:
    f.write('\n'.join(set(labels)))