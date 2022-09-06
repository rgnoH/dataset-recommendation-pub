import pandas as pd
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertTokenizer
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List
from tqdm import tqdm as tq
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import argparse
import os
import heapq


class SiameseBert(nn.Module):
    def __init__(self, model_name: str):
        super(SiameseBert, self).__init__()
        self.bert1 = BertModel.from_pretrained(model_name, local_files_only=True)
        # self.bert2 = deepcopy(self.bert1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        cls = self.bert1(x)[1]  # bs, 768
        pred = self.linear(self.relu(cls))[:, 0]  # bs,
        return self.sigmoid(pred)  # bs,


def collate_fn(batch_list: List[List[Tensor]]):
    pos_ids = [x[0] for x in batch_list]
    neg_ids = [x[1] for x in batch_list]
    return pad_sequence(pos_ids, batch_first=True), pad_sequence(neg_ids, batch_first=True)


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


@torch.no_grad()
def dev(model: SiameseBert, device: torch.device, dev_list: List[List[Tensor]]):
    model.eval()
    model.to(device)
    dev_loader = DataLoader(dataset=MyDataset(dev_list), batch_size=32, shuffle=False, collate_fn=collate_fn)
    answer = []
    for pos_ids, neg_ids in tq(dev_loader):
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)
        pos_pred = model(pos_ids).tolist()
        neg_pred = model(neg_ids).tolist()
        for i in range(len(pos_ids)):
            answer.append(pos_pred[i] > 0.5)
        for i in range(len(neg_ids)):
            answer.append(neg_pred[i] < 0.5)
    acc = sum(answer) / len(answer)
    print("Acc=", acc)
    return acc


def train(model: SiameseBert, device: torch.device, train_list: List[List[Tensor]],
          dev_list: List[List[Tensor]], optimizer: torch.optim.Optimizer,
          save_path: str, n_epochs: 50):
    model.to(device)
    loss_fn = nn.BCELoss()
    train_loader = DataLoader(dataset=MyDataset(train_list), batch_size=32, shuffle=True, collate_fn=collate_fn)
    best_acc = 0
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        losses = []
        model.train()
        for pos_ids, neg_ids in tq(train_loader):
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)
            optimizer.zero_grad()
            pos_pred = model(pos_ids)
            neg_pred = model(neg_ids)
            pos_label = torch.ones([len(pos_ids)], device=device, dtype=torch.float)
            neg_label = torch.zeros([len(neg_ids)], device=device, dtype=torch.float)
            loss = loss_fn(pos_pred, pos_label) + loss_fn(neg_pred, neg_label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Loss=", np.mean(losses))
        acc = dev(model, device, dev_list)
        if acc > best_acc:
            best_acc = acc
            print("Saving to", save_path)
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)


@torch.no_grad()
def test(model: SiameseBert, device: torch.device, test_list: List[List[int]]):
    test_loader = DataLoader(dataset=MyDataset(test_list), batch_size=32, shuffle=False, collate_fn=collate_fn)
    prob = []
    print("Testing")
    model.to(device)
    model.eval()
    for pos_ids, _ in tq(test_loader):
        pos_ids = pos_ids.to(device)
        pred = model(pos_ids).tolist()
        prob += pred
    index_prob = [(i, prob[i]) for i in range(len(prob))]
    heapq.heapify(index_prob)
    n_largest = heapq.nlargest(100000, index_prob, key=lambda x: x[1])
    n_indices = [x[0] for x in n_largest]
    return n_indices


def tokenize_df(df: pd.DataFrame, tokenizer: BertTokenizer):
    data_list = []
    for i, row in df.iterrows():
        context = row['context']
        research = row['research_question']
        random_q = row['random_question']
        pos_ids = tokenizer.encode(context, research, return_tensors="pt")[0]
        neg_ids = tokenizer.encode(research, random_q, return_tensors="pt")[0]
        data_list.append([pos_ids, neg_ids])
    return data_list


def optimizer_to(optim, device):
    """
    把 optimizer 对象送入指定device
    :param optim:
    :param device:
    :return:
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", default=2, type=int)
    random.seed(100)
    args = parser.parse_args()
    gpu_num = args.gpu_num
    save_path = "/home/xiajinxiong/workspace/bio/saved_model/data_filter.pt"
    df = pd.read_csv("/home/xiajinxiong/workspace/bio/data/prompt/questions.csv", sep="\t")
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=100), [10000, 20000])
    model_name = '/home/xiajinxiong/workspace/huggingface/bert-base-uncased'
    model = SiameseBert(model_name)
    if os.path.exists(save_path):
        print("Loading model from", save_path)
        model.load_state_dict(torch.load(save_path, map_location="cuda:" + str(gpu_num)))
    for p in model.parameters():
        p.requires_grad = True
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:" + str(gpu_num))
    train_list = tokenize_df(train_df, tokenizer)
    dev_list = tokenize_df(dev_df, tokenizer)
    test_list = tokenize_df(test_df, tokenizer)
    optimizer = Adam(params=model.parameters(), lr=1e-5)
    optimizer_to(optimizer, device)
    # train(model, device, train_list, dev_list, optimizer, save_path, 20)
    test_indices = test(model, device, test_list)
    filter_df = test_df.iloc[test_indices]
    filter_df.to_csv("/home/xiajinxiong/workspace/bio/data/prompt/filtered_data.csv")


if __name__ == '__main__':
    main()
