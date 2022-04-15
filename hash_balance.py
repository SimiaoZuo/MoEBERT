import argparse
import numpy as np
import pickle

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

np.random.seed(0)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_train_dataset(args):
    datasets = load_dataset("glue", args.task_name)

    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        preprocess_args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*preprocess_args, padding=padding, max_length=max_seq_length, truncation=True)
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not args.overwrite_cache)
    train_dataset = datasets["train"]
    return train_dataset, config


def get_balance_list(args):
    train_dataset, config = get_train_dataset(args)
    vocab_size = config.vocab_size

    print("Computing token frequency...")
    frequency = [0 for i in range(vocab_size)]
    for item in tqdm(train_dataset):
        item_ids = item["input_ids"]
        for num in item_ids:
            frequency[num] += 1
    frequency = np.array(frequency)

    frequency_sorted = np.sort(frequency)[::-1]
    frequency_ind = np.argsort(frequency)[::-1]

    print("Computing balanced hash list...")
    frequency_sorted = frequency_sorted.tolist()
    frequency_ind = frequency_ind.tolist()
    balance_list = [0 for i in range(vocab_size)]
    bucket_size = np.array([0 for i in range(args.num_buckets)])
    for freq, ind in tqdm(zip(frequency_sorted, frequency_ind)):
        if freq == 0:
            balance_list[ind] = np.random.randint(low=0, high=args.num_buckets)
        else:
            bucket_ind = np.argmin(bucket_size)
            bucket_size[bucket_ind] += freq
            balance_list[ind] = bucket_ind

    return balance_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    args = parser.parse_args()
    args.overwrite_cache = False
    args.pad_to_max_length = False
    args.tokenizer_name = None
    args.cache_dir = None
    args.use_fast_tokenizer = True
    args.model_revision = "main"
    args.use_auth_token = False

    balance_list = get_balance_list(args)
    name = "balance_hash_bucket_" + str(args.num_buckets) + "_" + str(args.task_name) + ".pkl"
    with open(name, "wb") as file:
        pickle.dump(balance_list, file)
    print("Completed!")


if __name__ == "__main__":
    main()
