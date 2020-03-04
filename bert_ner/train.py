# coding=utf8


import torch
import argparse
import os
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from bert_ner import dataset
from bert_ner.utils import file_helper
from bert_ner import model
from bert_ner.dataset import NerDataset
from torch.utils.data import dataloader, RandomSampler, DistributedSampler

glob_iters = 0


class Trainer(object):
    def __init__(self, args):
        self.local_rank = args.local_rank
        if args.local_rank == -1:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.n_gpu = 1

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        self.tokenizer = BertTokenizer.from_pretrained(os.getenv("BERT_BASE_CHINESE_VOCAB", "bert-base-chinese"),
                                                       do_lower_case=True)

        if args.model == "crf":
            self.model = model.NetCRF(args.top_rnns, len(NerDataset.VOCAB), self.device, args.finetuning,
                                      dropout=args.dropout, lin_dim=args.lin_dim)
        else:
            self.model = model.Net(args.top_rnns, len(NerDataset.VOCAB), self.device, args.finetuning,
                                   dropout=args.dropout, lin_dim=args.lin_dim)
        if args.local_rank == 0:
            torch.distributed.barrier()

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        elif args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                                   output_device=args.local_rank,
                                                                   find_unused_parameters=True)

        self.model.to(self.device)

    def train(self, train_path, valid_path, log_dir, model_dir, batch_size=32, sep="\t", epochs=30,
              lr=0.001, lr_decay_pat=3):
        best_model_path = os.path.join(model_dir, "best")
        file_helper.mk_folder_for_file(best_model_path)
        sm_writer = SummaryWriter(log_dir)
        train_loader = self._get_loader(train_path, batch_size, sep)
        valid_loader = self._get_loader(valid_path, batch_size, sep)
        # optimizer = Adam(self.model.parameters(), lr=lr)
        bert_parameters = list(map(id, self.model.bert.parameters()))
        rest_parameters = filter(lambda x: id(x) not in bert_parameters, self.model.parameters())
        optimizer = Adam([
            {"params": self.model.bert.parameters(),  "lr": 1e-5},
            {"params": rest_parameters, "lr": lr},
        ])
        best_f1 = 0.
        lr_decay_count = 0
        for epoch in tqdm(range(epochs)):
            self.train_epoch(train_loader, optimizer, sm_writer)
            print(f"=========test metric at epoch={epoch}=========")
            metric_info = self.evaluate(valid_loader)
            self._write_metric(sm_writer, metric_info, epoch, sign="test")
            _, _, f1 = metric_info
            print(f"=========train metric at epoch={epoch}=========")
            metric_info = self.evaluate(train_loader)
            self._write_metric(sm_writer, metric_info, epoch, sign="train")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), f"{best_model_path}.pt")
                lr_decay_count = 0
            else:
                lr_decay_count += 1
                if lr_decay_count == lr_decay_pat:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5
                        cur_lr = param_group["lr"]
                    lr_decay_count = 0
                    if cur_lr < 1e-8:
                        print(f"INFO: early stopping at {epoch}")
                        break

    def train_epoch(self, d_loader, optimizer, sm_writer):
        global glob_iters
        self.model.train()
        iter_ = iter(d_loader)
        i = 0
        while True:
            optimizer.zero_grad()
            try:
                batch = next(iter_)
                glob_iters += 1
                i += 1
                words, x, is_heads, tags, y, seqlens = dataset.trunk_batch(batch)
                logits, y_hat, loss = self.model(x, y)
                if torch.cuda.device_count() > 1:
                   loss = loss.mean()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"step: {i}, loss: {loss.item()}")
                    sm_writer.add_scalar('loss/train', loss.item(), glob_iters)
                    sm_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], glob_iters)
            except StopIteration:
                break

    def evaluate(self, d_loader):
        self.model.eval()
        iter_ = iter(d_loader)
        Y = []
        Y_hat = []
        with torch.no_grad():
            for i, batch in enumerate(iter_):
                words, x, is_heads, tags, y, seqlens = dataset.trunk_batch(batch)
                _, y_hat = self.model(x)
                self._save_to_total(seqlens, Y, y)
                self._save_to_total(seqlens, Y_hat, y_hat) 
        precision, recall, f1 = self._calc_metric(Y, Y_hat)
        print("precision=%.4f" % precision)
        print("recall=%.4f" % recall)
        print("f1=%.4f" % f1)
        return precision, recall, f1

    def _convert_label(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(y, list):
            y = np.array(y)
        return y.tolist()

    def _save_to_total(self, seqlens, total, cur_arrs):
        cur_arrs = self._convert_label(cur_arrs)
        for i, cur_arr in enumerate(cur_arrs):
            len_ = seqlens[i]
            total.extend(cur_arr[:len_]) 

    def _calc_metric(self, Y, Y_hat):
        Y = np.array(Y)
        Y_hat = np.array(Y_hat)
        print(type(Y_hat))
        print(Y_hat.shape)
        num_proposed = len(Y_hat[Y_hat > 1])
        num_correct = (np.logical_and(Y == Y_hat, Y > 1)).astype(np.int).sum()
        num_gold = len(Y[Y > 1])
        try:
            precision = num_correct / num_proposed
        except ZeroDivisionError:
            precision = 1.0

        try:
            recall = num_correct / num_gold
        except ZeroDivisionError:
            recall = 1.0

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            if precision * recall == 0:
                f1 = 1.0
            else:
                f1 = 0
        return precision, recall, f1

    def _get_loader(self, f_path, batch_size, sep="\t"):
        texts, labels = dataset.prepare_data(f_path, sep)
        ds = NerDataset(self.tokenizer, texts, labels)
        train_sampler = RandomSampler(ds) if self.local_rank == -1 else DistributedSampler(ds)
        d_loader = dataloader.DataLoader(dataset=ds, batch_size=batch_size,
                                         sampler=train_sampler, num_workers=4, collate_fn=dataset.pad)
        return d_loader

    def _write_metric(self, sm_writer, metric_info, epoch, sign="test"):
        precision, recall, f1 = metric_info
        sm_writer.add_scalar(f'precision/{sign}', precision, epoch)
        sm_writer.add_scalar(f'recall/{sign}', recall, epoch)
        sm_writer.add_scalar(f'f1/{sign}', f1, epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--model_dir", type=str, default=file_helper.get_data_file("models"))
    parser.add_argument("--trainset", type=str, default=None)
    parser.add_argument("--validset", type=str, default=None)
    parser.add_argument("--model", type=str, default="crf")
    parser.add_argument("--lr_decay_pat", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--lin_dim", type=int, default=128)
    parser.add_argument("--log_dir", type=str, default=file_helper.get_data_file("model_logs"),
                        help="path to save the logging of training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    return args


def main(args):
    trainer = Trainer(args)
    trainer.train(args.trainset, args.validset, args.log_dir, args.model_dir, args.batch_size, sep="\t",
                  epochs=args.epochs, lr=args.lr, lr_decay_pat=args.lr_decay_pat)


if __name__ == "__main__":
    t_args = parse_args()
    main(t_args)
