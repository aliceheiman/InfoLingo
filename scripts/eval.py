from utils import *
from datasets import load_dataset
import numpy as np

ds = load_dataset("rajpurkar/squad_v2")

n = 1000
dataset = ds["validation"]
gold_answers_map = get_gold_answers(dataset, n, empty=False)
qids = list(gold_answers_map.keys())

data = load_json("results/experiment_1732920760.json")
content = f"vocab_func,target_pct,accuracy_mean,accuracy_std,accuracy_se,f1_mean,f1_std,f1_se\n"
for method, items in data.items():
    for pct, answers in items.items():
        pred_answers_map = answers["predictions"]
        exact_scores, f1_scores = get_scores(gold_answers_map, pred_answers_map)
        eval_dict = make_eval_dict(exact_scores, f1_scores)

        exact_scores = list(exact_scores.values())
        f1_scores = list(f1_scores.values())

        accuracy_mean = np.mean(exact_scores)
        accuracy_std = np.std(exact_scores, ddof=1)

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores, ddof=1)

        # Calculate the number of runs/folds
        n = len(exact_scores)

        # Compute the standard error of the mean
        accuracy_se = accuracy_std / np.sqrt(n)
        f1_se = f1_std / np.sqrt(n)

        row = [
            f"{method}",
            f"{pct}",
            f"{(accuracy_mean * 100):.2f}",
            f"{(accuracy_std * 100):.2f}",
            f"{(accuracy_se * 100):.2f}",
            f"{(f1_mean * 100):.2f}",
            f"{(f1_std * 100):.2f}",
            f"{(f1_se * 100):.2f}",
        ]
        content += ",".join(row) + "\n"

with open("results/experiment_1732920760_extra_stats.csv", "w") as f:
    f.write(content)
