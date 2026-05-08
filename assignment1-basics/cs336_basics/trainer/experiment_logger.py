import csv
import json
import os
import pathlib
import re
import socket
from datetime import datetime


def slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


def infer_assignment_name():
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if parent.name.startswith("assignment"):
            return slugify(parent.name)
    return "cs336"


def infer_dataset_name(data_dir, data_meta):
    tokenizer_dir = data_meta.get("tokenizer_dir")
    if tokenizer_dir:
        return slugify(pathlib.Path(tokenizer_dir).name)
    return slugify(pathlib.Path(data_dir).name)


def build_run_name(args, assignment_name, dataset_name):
    if args.wandb_run_name:
        return args.wandb_run_name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = slugify(socket.gethostname())
    model_part = f"l{args.num_layers}-d{args.d_model}-h{args.num_heads}-ctx{args.context_len}"
    train_part = f"bs{args.batch_size}-steps{args.train_steps}-lr{args.max_lr:g}"
    return f"{assignment_name}-lm-{dataset_name}-{model_part}-{train_part}-{host}-{timestamp}"


def get_model_config(args):
    return {
        "vocab_size": args.vocab_size,
        "context_length": args.context_len,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "rope_theta": args.rope_theta,
        "d_ff": args.d_ff,
    }


def get_optimizer_config(args):
    return {
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "warm_up_it": args.warm_up_it,
        "cosine_it": args.cosine_it,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "clip_grad_norm": args.clip_grad_norm,
    }


def get_training_config(args):
    return {
        "batch_size": args.batch_size,
        "train_steps": args.train_steps,
        "val_interval": args.val_interval,
        "val_batches": args.val_batches,
        "save_intervals": args.save_intervals,
        "log_intervals": args.log_intervals,
        "resume_ckp": args.resume_ckp,
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class LocalExperimentLogger:
    def __init__(self, args, device, data_meta, model_config):
        self.disabled = args.no_local_log
        self.latest_train = None
        self.latest_val = None

        if self.disabled:
            return

        self.save_dir = os.path.abspath(args.save_ckp_path)
        os.makedirs(self.save_dir, exist_ok=True)

        self.started_at = datetime.now().isoformat()
        self.assignment_name = infer_assignment_name()
        self.dataset_name = infer_dataset_name(args.data_dir, data_meta)
        self.run_name = build_run_name(args, self.assignment_name, self.dataset_name)
        self.config_path = os.path.join(self.save_dir, "config.json")
        self.metrics_path = os.path.join(self.save_dir, "metrics.jsonl")
        self.metrics_csv_path = os.path.join(self.save_dir, "metrics.csv")
        self.summary_path = os.path.join(self.save_dir, "summary.json")

        config = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "host": socket.gethostname(),
            "assignment": self.assignment_name,
            "dataset_name": self.dataset_name,
            "device_requested": args.device,
            "device_resolved": device,
            "data_dir": args.data_dir,
            "save_ckp_path": self.save_dir,
            "model_config": model_config,
            "optimizer_config": get_optimizer_config(args),
            "training_config": get_training_config(args),
            "wandb": {
                "enabled": not args.no_wandb,
                "entity": args.wandb_entity,
                "project": args.wandb_project,
                "group": args.wandb_group,
                "job_type": args.wandb_job_type,
                "run_name": args.wandb_run_name,
                "tags": args.wandb_tags,
            },
            "tokenized_data_meta": data_meta,
            "all_args": vars(args),
        }
        write_json(self.config_path, config)

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            f.write("")
        with open(self.metrics_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")

        print(f"Local run config: {self.config_path}")
        print(f"Local metrics log: {self.metrics_path}")
        print(f"Local metrics csv: {self.metrics_csv_path}")

    def add_dataset_info(self, train_data_path, val_data_path, train_tokens, val_tokens, total_params):
        if self.disabled:
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["dataset"] = {
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens),
        }
        config["model_total_parameters"] = int(total_params)
        write_json(self.config_path, config)

    def log_metric(self, record):
        if self.disabled:
            return

        record = {
            "time": datetime.now().isoformat(),
            **record,
        }
        self._append_jsonl(record)
        self._append_csv(record)

        if record.get("split") == "train":
            self.latest_train = record
        elif record.get("split") == "val":
            self.latest_val = record

    def write_summary(self, completed_iter, final_checkpoint_path, total_params, status, elapsed_seconds):
        if self.disabled:
            return

        summary = {
            "run_name": self.run_name,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_iter": completed_iter,
            "total_params": int(total_params),
            "final_checkpoint_path": final_checkpoint_path,
            "latest_train": self.latest_train,
            "latest_val": self.latest_val,
            "metrics_path": self.metrics_path,
            "metrics_csv_path": self.metrics_csv_path,
            "config_path": self.config_path,
        }
        write_json(self.summary_path, summary)
        print(f"Local summary: {self.summary_path}")

    def _append_jsonl(self, record):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_csv(self, record):
        fieldnames = [
            "time",
            "split",
            "iteration",
            "loss",
            "avg_loss",
            "perplexity",
            "learning_rate",
            "val_batches",
        ]
        file_exists = os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0
        with open(self.metrics_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: record.get(key) for key in fieldnames})
