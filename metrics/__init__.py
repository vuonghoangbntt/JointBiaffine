from metrics.sequence_labeling import *
import os
import matplotlib.pyplot as plt


def batch_computeF1(intent_labels, intent_preds, slot_labels, slot_preds, seq_lengths, label_set, save_path=None,
                    do_error_analyze=False, samples=None, intent_maps=None):
    y_true = []
    y_pred = []
    sent_true = 0
    intent_accuracy = np.sum(np.array(intent_labels) == np.array(intent_preds)) / len(intent_labels)
    intent_count = {}
    with open(os.path.join(save_path, "error_analysis.txt"), 'w') as f:
        for i in range(len(slot_labels)):
            intent_label = intent_labels[i]
            intent_pred = intent_preds[i]
            slot_label = slot_labels[i]
            slot_pred = slot_preds[i]

            true_len = seq_lengths[i].item()
            pred = slot_pred[:true_len, :true_len]
            label = slot_label[:true_len, :true_len]
            predict_entity, label_entity = get_entities(pred, label, label_set)
            y_true.append(label_entity)
            y_pred.append(predict_entity)
            flag = 1
            label_entity = sorted(list(label_entity), key=lambda x: x[1])
            predict_entity = sorted(list(predict_entity), key=lambda x: x[1])
            if intent_label != intent_pred:
                flag = 0
                if intent_maps[int(intent_label)] not in intent_count:
                    intent_count[intent_maps[int(intent_label)]] = 1
                else:
                    intent_count[intent_maps[int(intent_label)]] += 1
            if len(label_entity) != len(predict_entity):
                flag = 0
            else:
                for j in range(len(label_entity)):
                    if label_entity[j][0] != predict_entity[j][0] \
                            or label_entity[j][1] != predict_entity[j][1] \
                            or label_entity[j][2] != predict_entity[j][2]:
                        flag = 0
                        break
            if flag == 0 and do_error_analyze:
                f.write(' '.join(samples[i]['sentence']))
                f.write('\n' + str(intent_maps[int(intent_label)]))
                f.write('\n' + str(label_entity))
                f.write('\n' + str(intent_maps[int(intent_pred)]))
                f.write('\n' + str(predict_entity))
                f.write('\n')
                f.write('---------------------------------------\n')
            sent_true += flag
    if do_error_analyze:
        plot_chart(intent_count, os.path.join(save_path, "intent_analyze.png"))
        with open(os.path.join(save_path, "slot_analyze.txt"), "w") as f:
          f.write(classification_report(y_true, y_pred, digits=4))
        with open(os.path.join(save_path, "intent_analyze.txt"), "w") as f:
          f.write(str(intent_count))
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true,
                                                                                   y_pred), classification_report(
        y_true, y_pred, digits=4), intent_accuracy, float(sent_true) / len(intent_labels)


def get_entities(input_tensor, label, label_set):
    input_tensor, cate_pred = input_tensor.max(dim=-1)
    predict_entity = get_pred_entity(cate_pred, input_tensor, label_set, True)
    label_entity = get_entity(label, label_set)
    return predict_entity, label_entity


def get_pred_entity(cate_pred, span_scores, label_set, is_flat_ner=True):
    top_span = []
    for i in range(len(cate_pred)):
        for j in range(i, len(cate_pred)):
            if cate_pred[i][j] > 0:
                tmp = (label_set[cate_pred[i][j].item()], i, j, span_scores[i][j].item())
                top_span.append(tmp)
    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    res_entity = []
    for t, ns, ne, _ in top_span:
        for _, ts, te, in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are not allowed
                break
        else:
            res_entity.append((t, ns, ne))
    return set(res_entity)


def get_entity(input_tensor, label_set):
    entity = []
    for i in range(len(input_tensor)):
        for j in range(i, len(input_tensor)):
            if input_tensor[i][j] > 0:
                tmp = (label_set[input_tensor[i][j].item()], i, j)
                entity.append(tmp)
    return entity


def plot_chart(analytics, save_path=None):
    left = [2*i for i in range(len(analytics))]
    height = [analytics[key] for key in analytics]
    tick_label = [key for key in analytics]
    plt.bar(left, height, tick_label=tick_label, width=1.2, color=['red', 'green'])
    # naming the x-axis
    plt.xlabel('Label')
    # naming the y-axis
    plt.ylabel('Wrong')
    # plot title
    plt.title('Error Analysis')

    if save_path is not None:
        plt.savefig(save_path)
    # function to show the plot
    plt.show()