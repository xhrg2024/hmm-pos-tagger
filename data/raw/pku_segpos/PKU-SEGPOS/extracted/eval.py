import json
import argparse
import sys


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()
    return data


def examine_duplicate(lst):
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j]:
                raise RuntimeError("duplicate prediction exists")


def f1_metric(golden_file, pred_file, return_dict=True):
    golden_data = load_file(golden_file)
    pred_data = load_file(pred_file)

    if len(golden_data) != len(pred_data):
        raise RuntimeError("Wrong Predictions")

    pred_correct = 0
    total_golden = 0
    total_pred = 0

    for golden_sent, pred_sent in zip(golden_data, pred_data):
        assert "span_list" in golden_sent
        assert "span_list" in pred_sent

        golden_spans = golden_sent["span_list"]
        pred_spans = pred_sent["span_list"]
        total_golden += len(golden_spans)
        total_pred += len(pred_spans)

        examine_duplicate(golden_spans)
        examine_duplicate(pred_spans)
        for golden_entity in golden_spans:
            for pred_entity in pred_spans:
                if golden_entity == pred_entity:
                    pred_correct += 1
                    break

    precision = pred_correct / total_pred if total_pred > 0 else 0
    recall = pred_correct / total_golden if total_golden > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if return_dict:
        return {"precision": precision, "recall": recall, "f1": f1,
                "pred_correct": pred_correct, "total_pred": total_pred, "total_golden": total_golden}
    else:
        return precision, recall, f1


def main():
    argv = sys.argv
    print("预测结果：{}, 测试集: {}".format(argv[1], argv[2]))
    print(f1_metric(argv[2], argv[1]))


if __name__ == '__main__':
    main()
