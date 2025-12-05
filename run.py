
import cs336_basics
import json
import argparse
import numpy as np
import random
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm
import os
from pathlib import Path
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
import math
import logging
from contextlib import nullcontext

device = torch.cuda.is_available() and "cuda:0" or "cpu"
logger = logging.getLogger(__name__)


def nvtx_range(name: str):
    # Use NVTX ranges when CUDA is available; otherwise, no-op.
    return nvtx.range(name) if torch.cuda.is_available() else nullcontext()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        type=str,
                        choices=["tokenizer", "train", "infer"],
                        help="Mode to run: tokenizer, train, or infer")
    parser.add_argument(
        "--config", type=str, default="config/config.json", help="Path to the config file")
    parser.add_argument(
        "--profile", action="store_true", help="Enable short profiling run")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed")
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


def set_seed(seed: int | None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tokenizer(config: dict):
    dataset_name = config["data"]
    train_txt_path = config[dataset_name]["train_txt_path"]
    train_bin_path = config[dataset_name]["train_bin_path"]
    valid_txt_path = config[dataset_name]["valid_txt_path"]
    valid_bin_path = config[dataset_name]["valid_bin_path"]
    bpe_model_path = config[dataset_name]["bpe_model_path"]
    vocab_size = config["vocab_size"]
    special_tokens = config["special_tokens"]

    if Path(bpe_model_path).exists():
        msg = f"BPE model already exists at {bpe_model_path}, loading without retraining."
        logger.info(msg)
        print(msg)
        vocab, merges = cs336_basics.tokenizer.load_bpe_model(bpe_model_path)
        logger.info("BPE model loaded successfully.")
        print("BPE model loaded successfully.")
    else:
        logger.info(f"Training tokenizer on {train_txt_path}...")
        vocab, merges = cs336_basics.tokenizer.train_bpe(
            train_txt_path, vocab_size, special_tokens=special_tokens)
        cs336_basics.tokenizer.save_bpe_model(
            vocab, merges, bpe_model_path)
        new_vocab, new_merges = cs336_basics.tokenizer.load_bpe_model(
            bpe_model_path)
        assert vocab == new_vocab and merges == new_merges, "Loaded model does not match saved model!"
        logger.info("BPE model trained and saved successfully.")

    tokenizer = cs336_basics.tokenizer.Tokenizer(
        vocab, merges, special_tokens=special_tokens)

    if Path(train_bin_path).exists():
        msg = f"Tokenized train file already exists at {train_bin_path}, skipping."
        logger.info(msg)
        print(msg)
    else:
        logger.info(f"Tokenizing {train_txt_path}...")
        tokenizer.encode_file2file(train_txt_path, train_bin_path)

    if Path(valid_bin_path).exists():
        msg = f"Tokenized valid file already exists at {valid_bin_path}, skipping."
        logger.info(msg)
        print(msg)
    else:
        logger.info(f"Tokenizing {valid_txt_path}...")
        tokenizer.encode_file2file(valid_txt_path, valid_bin_path)


def train(config: dict, profile: bool = False):
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
        gradient_clipping=config["gradient_clipping"],
    )

    # Load model
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        start_iter = 1 + cs336_basics.nn.utils.load_checkpoint(
            model, optimizer, checkpoint_path
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Create new model...")
        logger.info(f"Create new model...")
        start_iter = 0

    # DataLoader
    print(f"Prepare dataset/dataloader")
    logger.info(f"Prepare dataset/dataloader")
    memmap = np.memmap(train_bin_path, dtype=np.uint32, mode="r")
    dataset = cs336_basics.CS336Dataset(memmap, context_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader_iter = iter(loader)

    # Profiling
    profile_begin = 30
    profile_window = 50

    # Training
    # skip start_iter batch from checkpoint
    for _ in range(start_iter):
        next(loader_iter)
    num_of_iters = len(dataset) // (batch_size * context_length)
    postfix = {}
    print(f"Training start")
    with tqdm(total=num_of_iters, initial=start_iter, desc="iterations") as pbar:
        it = start_iter
        for x_cpu, y_cpu in loader_iter:
            if profile and it == profile_begin:
                nvtx.range_push(f"profile_window_{it}")
            if profile and it == profile_begin + profile_window:
                nvtx.range_pop()
                break
            with nvtx_range(f"data_batch_{it}"):
                x = x_cpu.to(device, dtype=torch.long, non_blocking=True)
                y = y_cpu.to(device, dtype=torch.long, non_blocking=True)
            with nvtx_range(f"forward_{it}"):
                logits = model(x)
                loss = cs336_basics.nn.function.cross_entropy_loss(
                    logits=logits, target=y)
            with nvtx_range(f"backward_{it}"):
                optimizer.zero_grad()
                loss.backward()
            with nvtx_range(f"optimizer_step_{it}"):
                optimizer.step()

            it += 1
            if it == num_of_iters:
                break
            if it % 10 == 0:
                postfix["loss"] = f"{loss.item():.3f}"
                logger.info(f"Iter {it}: {postfix}")
                pbar.set_postfix(postfix, refresh=False)
            pbar.update(1)
            # Save model every 200 iters
            if it % 200 == 0:
                with nvtx_range(f"save_checkpoint_{it}"):
                    cs336_basics.nn.utils.save_checkpoint(
                        model, optimizer, it, checkpoint_path)
                with nvtx_range(f"eval_{it}"):
                    eval_result = eval(config, model)
                postfix["eval_loss"] = f"{eval_result['loss']:.3f}"
                postfix["acc"] = f"{eval_result['accuracy']:.3f}"
                postfix["perplexity"] = f"{eval_result['perplexity']:.3f}"
            if it % 1000 == 0:
                with nvtx_range(f"save_checkpoint_extra_{it}"):
                    cs336_basics.nn.utils.save_checkpoint(
                        model, optimizer, it, checkpoint_path.with_name(f"mode_{it + 1}.pt"))


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

    # Seed
    set_seed(args.seed)

    if args.mode == "tokenizer":
        tokenizer(config)
    elif args.mode == "train":
        train(config, profile=args.profile)
    elif args.mode == "infer":
        raise ValueError("Not implemented yet.")


if __name__ == "__main__":
    main()
