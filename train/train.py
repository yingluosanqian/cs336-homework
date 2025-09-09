
import cs336_basics
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import os
from pathlib import Path
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
import math
import logging

device = torch.cuda.is_available() and "cuda:0" or "cpu"
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        type=str,
                        choices=["data", "train", "infer"],
                        help="Mode to run: data, train, or infer")
    parser.add_argument(
        "--config", type=str, default="train/train.json", help="Path to the config file")
    args = parser.parse_args()
    return args


def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def init(config):
    log_path = Path(config["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            # logging.StreamHandler()
        ]
    )


def train(config: dict):
    dataset_name = config["data"]
    train_bin_path = config[dataset_name]["train_bin_path"]
    batch_size = config["batch_size"]
    context_length = config["context_length"]

    # Transformer
    model = cs336_basics.nn.basic.TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )

    # Optimizer
    optimizer = cs336_basics.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
    )

    # Load model
    checkpoint_path = config["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        start_iter = 1 + cs336_basics.nn.utils.load_checkpoint(
            model, optimizer, checkpoint_path
        )
    else:
        start_iter = 0

    data = np.memmap(train_bin_path, dtype=np.uint32, mode="r")
    num_of_iters = (len(data) + (batch_size * context_length) -
                    1) // (batch_size * context_length)
    postfix = {}
    with tqdm(range(start_iter, num_of_iters), desc="iterations") as pbar:
        for iter in range(start_iter, num_of_iters):
            x, y = cs336_basics.nn.utils.get_batch(
                data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            loss = cs336_basics.nn.function.cross_entropy_loss(
                logits=logits, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            postfix["loss"] = f"{loss.item():.3f}"
            # Save model every 100 iters
            if (iter + 1) % 100 == 0:
                cs336_basics.nn.utils.save_checkpoint(
                    model, optimizer, iter, checkpoint_path)
                eval_result = eval(config, model)
                postfix["eval_loss"] = f"{eval_result['loss']:.3f}"
                postfix["acc"] = f"{eval_result['accuracy']:.3f}"
                postfix["perplexity"] = f"{eval_result['perplexity']:.3f}"
            logger.info(f"Iter {iter}: {postfix}")
            pbar.set_postfix(postfix)
            pbar.update(1)


def eval(config: dict,
         model: cs336_basics.nn.basic.TransformerLM):
    with torch.no_grad():
        dataset_name = config["data"]
        valid_bin_path = config[dataset_name]["valid_bin_path"]
        batch_size = config["batch_size"]
        context_length = config["context_length"]

        data = np.memmap(valid_bin_path, dtype=np.uint32, mode="r")
        # Only eval on 1/4 of the valid set for speed
        num_of_iters = (len(data) + (batch_size * context_length) -
                        1) // (batch_size * context_length) // 4
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        for iter in range(num_of_iters):
            x, y = cs336_basics.nn.utils.get_batch(
                data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            x: Int[Tensor, "batch context_length"]
            y: Int[Tensor, "batch context_length"]
            logits: Int[Tensor, "batch context_length vocab_size"] = model(x)
            loss = cs336_basics.nn.function.cross_entropy_loss(
                logits=logits, target=y
            )
            total_loss += loss.item()
            output: Int[Tensor, "batch context_length"] = logits.argmax(dim=-1)
            correct = (output == y).sum().item()
            total_correct += correct
            total_tokens += y.numel()
        acc = total_correct / total_tokens
        return {
            "loss": total_loss / num_of_iters,
            "accuracy": acc,
            "perplexity": math.exp(total_loss / num_of_iters)
        }


def main():
    # Args
    args = parse_args()

    # Config
    config = get_config(args.config)

    # Initialization
    init(config)

    if args.mode == "data":
        dataset_name = config["data"]
        train_txt_path = config[dataset_name]["train_txt_path"]
        train_bin_path = config[dataset_name]["train_bin_path"]
        valid_txt_path = config[dataset_name]["valid_txt_path"]
        valid_bin_path = config[dataset_name]["valid_bin_path"]
        vocab_size = config["vocab_size"]
        special_tokens = config["special_tokens"]

        logger.info(f"Training tokenizer on {train_txt_path}...")
        vocab, merges = cs336_basics.tokenizer.train_bpe(
            train_txt_path, vocab_size, special_tokens=special_tokens)
        cs336_basics.tokenizer.save_bpe_model(
            vocab, merges, config[dataset_name]["bpe_model_path"])

        logger.info(f"Tokenizing {train_txt_path}...")
        tokenizer = cs336_basics.tokenizer.Tokenizer(
            vocab, merges, special_tokens=special_tokens)
        tokenizer.encode_file2file(train_txt_path, train_bin_path)

        logger.info(f"Tokenizing {valid_txt_path}...")
        tokenizer = cs336_basics.tokenizer.Tokenizer(
            vocab, merges, special_tokens=special_tokens)
        tokenizer.encode_file2file(valid_txt_path, valid_bin_path)

        # new_vocab, new_merges = cs336_basics.tokenizer.load_bpe_model(
        #     config[dataset_name]["bpe_model_path"])
        # assert vocab == new_vocab and merges == new_merges, "Loaded model does not match saved model!"
    elif args.mode == "train":
        train(config)
    elif args.mode == "infer":
        raise ValueError("Not implemented yet.")


if __name__ == "__main__":
    main()
