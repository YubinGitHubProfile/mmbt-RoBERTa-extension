#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from mmbt.data.helpers import get_data_loaders
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *

import matplotlib.pyplot as plt
import numpy as np
import math
from transformers import AdamW, RobertaTokenizer


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=32) #default: 128, change it to 16 to test
    parser.add_argument("--bert_model", type=str, default="roberta-base", choices=["bert-base-uncased", "bert-large-uncased", "roberta-base"])
    parser.add_argument("--data_path", type=str, default="path/to/hateful/memes/dataset")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=10) # default: 0
    parser.add_argument("--freeze_txt", type=int, default=10) # default: 0
    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt") # this is not need for training MMBT
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24) # default: 24
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4) #default: 1e-4
    parser.add_argument("--lr_factor", type=float, default=0.5) # default is 0.5 but it reduced the lr too quickly/agressively, change it to 0.2 or lower to see.
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=25)  #default 100, change it to 5 to test
    parser.add_argument("--max_seq_len", type=int, default=128) #default:512, for shorter one maybe choose 128
    parser.add_argument("--model", type=str, default="mmbt", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=2) #default: 12
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="path/to/checkpoint")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="meme", choices=["mmimdb", "vsnli", "food101", "meme"]) #added the meme dataset
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)



import json
#####added##############
prefix = 'path/to/the/hateful/memes/dataset'
train_data_path = prefix + "train.jsonl"
augmented_data_path = "path/to/train.jsonl" # whether there is an augmented dataset or not, this will work for both situations

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl(train_data_path)
train_size = len(train_data)
augmented_data = load_jsonl(augmented_data_path)
augmented_size = len(augmented_data)

#### defining these functions for train
def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = int(math.ceil(
            (augmented_size + 10)          #change it if no augmentation
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        ))
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.1}, # default: 0.01
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )

        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     lr=args.lr,
        #     eps=1e-8
        # )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        metrics["aucroc"] = roc_auc_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch):
    txt, segment, mask, img, tgt = batch

    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
    else:
        assert args.model == "mmbt"
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        # if args.bert_model == "roberta-base":
        out = model(txt, mask, segment, img) # Pass only txt, mask, and img for roberta-base
        # else:
        #   out = model(txt, mask, segment, img)

    tgt = tgt.cuda()
    loss = criterion(out, tgt)
    return loss, out, tgt


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")

    # To store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for i_epoch in range(start_epoch, args.max_epochs):
        epoch_train_losses = []
        epoch_train_correct = 0
        epoch_train_total = 0

        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, preds, labels = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate training accuracy
            _, predicted = torch.max(preds, 1)
            epoch_train_correct += (predicted == labels).sum().item()
            epoch_train_total += labels.size(0)

        # Calculate training loss and accuracy
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_accuracy = epoch_train_correct / epoch_train_total

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation phase
        model.eval()
        val_metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        val_loss = val_metrics["loss"]  # Assuming loss is part of the metrics
        val_accuracy = val_metrics["acc"]  # Assuming accuracy is part of the metrics
        val_aucroc = val_metrics["aucroc"]

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(f"Epoch {i_epoch}: Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")
        log_metrics("Val", val_metrics, args, logger)
        logger.info(f"AUC-ROC: {val_aucroc}")
        tuning_metric = (
            val_metrics["micro_f1"] if args.task_type == "multilabel" else val_metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    # After training, plot the loss and accuracy curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")

    plt.tight_layout()
    plt.show()

##########NEW script for training a RoBERTa architecture########
def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)

    # change
    args, remaining_args = parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])
    # args, remaining_args = parser.parse_known_args()
    # assert remaining_args == [], remaining_args
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)
    train(args)

# original script for train
# def cli_main():
#     parser = argparse.ArgumentParser(description="Train Models")
#     get_args(parser)
#     args, remaining_args = parser.parse_known_args()
#     assert remaining_args == [], remaining_args
#     train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
