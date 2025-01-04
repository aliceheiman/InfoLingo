import time

# Setup
timestamp = str(int(time.time()))
output_csv = f"results/experiment_{timestamp}.csv"
output_json = f"results/experiment_{timestamp}.json"

n = 1000
dataset = ds["validation"]
gold_answers_map = get_gold_answers(dataset, n, empty=False)
qids = list(gold_answers_map.keys())
relevant_dataset = [item for item in dataset if item["id"] in qids]

print("Num gold answers:", len(gold_answers_map))
assert set(gold_answers_map.keys()) == set([item["id"] for item in relevant_dataset])

prior_vocab_name = "A1A2"
prior_vocab = load_vocab(f"vocab/{prior_vocab_name}.txt")
# prior_vocab_name = "none"
# prior_vocab = []

contents = ["n,prior_vocab,vocab_func,target_pct,accuracy,f1,avg_vocab_len"]  # header
json_predictions = {}

# Run through baseline
vocab_func = pick_kl_divergence
vocab_name = vocab_func.__name__
json_predictions = load_json(output_json)
json_predictions[vocab_name] = {}
print(f"[*] Running experiments for {vocab_name}...")
#for target_pct in [0, 0.10, 0.25, 0.5, 0.75, 1.0]: # first run include none and all
for target_pct in [0.10, 0.25, 0.5, 0.75]:
    json_predictions[vocab_name][f"{target_pct}"] = {}
    pred_answers_map, vocabs = take_test(
        relevant_dataset,
        vocab_func=vocab_func,
        target_pct=target_pct,
        prior_vocab=prior_vocab
    )
    json_predictions[vocab_name][f"{target_pct}"]["predictions"] = pred_answers_map
    #json_predictions[vocab_name][f"{target_pct}"]["vocabs"] = vocabs

    exact_scores, f1_scores = get_scores(gold_answers_map, pred_answers_map)
    eval_dict = make_eval_dict(exact_scores, f1_scores)

    lens = [len(s) for s in vocabs.values()]
    avg_len = sum(lens) / len(lens)
    content = f"{n},{prior_vocab_name},{vocab_name},{target_pct},{eval_dict['exact']:.2f},{eval_dict['f1']:.2f},{avg_len:.2f}"
    print(f"\t{content}")
    contents.append(content)
    save_result(contents, output_csv)
    save_json(json_predictions, output_json)