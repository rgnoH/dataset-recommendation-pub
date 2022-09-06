import torch
import torch.nn as nn
from tqdm import tqdm as tq
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score

'''Model'''

class Model(nn.Module):
    # use matrix multiplication to accelerate computation of cosine similarity
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.mode = 'default'
        self.dic = {}  # store embedding of items
        self.matrix = None
        self.matrix_row_labels = []

    def forward(self, x, ids):
        hidden = self.bert(x)
        embed = hidden[1]
        if self.mode == 'store':
            for id, emb in zip(ids, embed):
                self.dic[id] = emb  # store embedding of items
        return embed

    def construct_matrix(self):
        # always called before calling match(x, expected_id)
        temp = []
        self.matrix_row_labels.clear()
        for id, embed in self.dic.items():
            self.matrix_row_labels.append(id)
            temp.append((embed / torch.norm(embed)).cpu().numpy())
        self.matrix = torch.tensor(temp).to(device)

    def match(self, x, expected_id):
        # print(f'start matching. {len(self.dic)} items in dic')
        x = x / torch.norm(x)
        cosine_sim = self.matrix @ x.reshape(-1, 1)
        lst = list(zip(cosine_sim.reshape(-1).tolist(), self.matrix_row_labels))
        lst.sort(key=lambda x: x[0], reverse=True)
        lst_n = [x[1] for x in lst]
        rets = []
        if lst_n[0] == expected_id:
            rets.append(1)
        else:
            rets.append(0)
        for k in PatK_Ks:
            if expected_id in lst_n[:k]:
                rets.append(1)
            else:
                rets.append(0)
        return rets

'''settings & hyper paramters'''

gpu_num = 3
device = torch.device("cuda:" + str(gpu_num))
train_batch_size = 32
test_batch_size = 64
n_epochs = 10
# model_name = '/home/xiajinxiong/workspace/bio/tanghaihong/model/bert-base-uncased'
model_name = 'bert-base-uncased'
model = Model()
train_pos_loader = None
train_neg_loader = None
dev_loader = None
test_loader = None
tokenizer = BertTokenizer.from_pretrained(model_name)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MarginRankingLoss()

class MyDataset(Dataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dic
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

def tokenize(df:pd.DataFrame):
    global tokenizer, cnt_tokenizer
    ret = []
    for i, row in tq(df.iterrows()):
        # row: question, abstract, summary, gse, id
        user = tokenizer.encode(row[0])[:256]
        item = tokenizer.encode(row[1])[:128] + tokenizer.encode('[SEP]') + tokenizer.encode(row[2])[:128]
        ret.append({'user': user, 'item': item, 'id': row[4], 'label': 1})
    return ret

def collate_fn(batch):
    global tokenizer
    padding_id = tokenizer.pad_token_id
    users = [torch.tensor(x['user'], dtype=torch.long)[:64] for x in batch]
    users = pad_sequence(users, batch_first=True, padding_value=padding_id)
    items = [torch.tensor(x['item'], dtype=torch.long)[:257] for x in batch]
    items = pad_sequence(items, batch_first=True, padding_value=padding_id)
    ids = [x['id'] for x in batch]
    labels = [x['label'] for x in batch]
    return users, items, ids, labels

def negative_sample(lst):
    # how to coin negative sample:
    # 1.shuffle items randomly
    # 2.handle fixed points: find the fixed points, and then construct a cycle out of them
    keys = [(x['user'], x['id']) for x in lst]
    values = [(x['item'], x['id']) for x in lst]
    random.shuffle(values)
    fixed = []
    for i in range(len(lst)):
        if keys[i][1] == values[i][1]:
            fixed.append(i)
    for i in range(len(fixed) - 1):
        j, k = fixed[i], fixed[i + 1]
        values[j], values[k] = values[k], values[j]
    return [{'user': k[0], 'item': v[0], 'id': v[1], 'label': 0} for k, v in zip(keys, values)]

def load_data():
    global train_pos_loader, train_neg_loader, dev_loader, test_loader
    train_df = pd.read_csv('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/sample/train_80000.csv')
    dev_df = pd.read_csv('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/sample/dev_9473.csv')
    test_df = pd.read_csv('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/sample/test_9473.csv')
    train_list = tokenize(train_df)
    train_neg_list = negative_sample(train_list)
    dev_list = tokenize(dev_df)
    dev_neg_list = negative_sample(dev_list)
    test_list = tokenize(test_df)
    test_neg_list = negative_sample(test_list)
    train_pos_loader = DataLoader(dataset=MyDataset(train_list), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
    train_neg_loader = DataLoader(dataset=MyDataset(train_neg_list), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=MyDataset(dev_list + dev_neg_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=MyDataset(test_list + test_neg_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)

def run_model(loader:DataLoader):
    model.eval()
    model.mode = 'store'
    model.dic.clear()
    print('start getting embeddings of items')
    for data in tq(loader):  # store item embedding of test dataset first
        items, ids = data[1], data[2]
        items = items.to(device)
        with torch.no_grad():
            _ = model(items, ids)
    model.construct_matrix()    # accelerate matching
    print('start running model')
    model.mode = 'default'
    preds = [[] for _ in range(len(PatK_Ks) + 1)]
    true = []
    for data in tq(loader):
        users, ids, labels = data[0], data[2], data[3]
        users = users.to(device)
        with torch.no_grad():
            users = model(users, ids)
            match_rets = [model.match(x, id) for x, id in zip(users, ids)]
            for i in range(len(preds)):
                preds[i] += [x[i] for x in match_rets]
        true += labels
    return preds, true

best_auc = 0.0
best_f1 = 0.0
best_acc = 0.0
PatK_Ks = [5, 10, 20, 50, 100]
best_PatKs = [0.0 for _ in range(len(PatK_Ks))]

def PatK_score(true, pred):
    tot = 0
    cnt = 0
    for t, p in zip(true, pred):
        if t == 1:
            tot += 1
            if p == 1:
                cnt += 1
    return cnt / tot

def validate():
    print('test preparing')
    preds, true = run_model(dev_loader)
    global best_auc, best_f1, best_acc, best_PatKs, PatK_Ks
    auc = roc_auc_score(true, preds[0])
    if auc > best_auc:
        torch.save(model.state_dict(), "/home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
                                       "/baseline_auc.pt", _use_new_zipfile_serialization=False)
        print("saving baseline to /home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
              "/baseline_auc.pt")
        best_auc = auc
    f1 = f1_score(true, preds[0])
    if f1 > best_f1:
        torch.save(model.state_dict(), "/home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
                                       "/baseline_f1.pt", _use_new_zipfile_serialization=False)
        print("saving baseline to /home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
              "/baseline_f1.pt")
        best_f1 = f1
    acc = accuracy_score(true, preds[0])
    if acc > best_acc:
        torch.save(model.state_dict(), "/home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
                                       "/baseline_acc.pt", _use_new_zipfile_serialization=False)
        print("saving baseline to /home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
              "/baseline_acc.pt")
        best_acc = acc
    for i in range(1, len(preds)):
        PatK = PatK_score(true, preds[i])
        if PatK > best_PatKs[i - 1]:
            torch.save(model.state_dict(), f"/home/xiajinxiong/workspace/bio/tanghaihong/temp_model"
                                           f"/baseline_PatK{PatK_Ks[i - 1]}.pt", _use_new_zipfile_serialization=False)
            print(f'saving baseline to /home/xiajinxiong/workspace/bio/tanghaihong/temp_model'
                  f'/baseline_PatK{PatK_Ks[i - 1]}.pt')
            best_PatKs[i - 1] = PatK

def train():
    global model, train_pos_loader, train_neg_loader
    global best_auc, best_f1, best_acc, best_PatKs, PatK_Ks
    print("Training")
    losses = []
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        model.mode = 'default'
        print(f'epoch {epoch}:')
        for pos, neg in tq(zip(train_pos_loader, train_neg_loader)):
            optimizer.zero_grad()
            users, items, ids = pos[0], pos[1], pos[2]
            users = users.to(device)
            items = items.to(device)
            user_embed = model(users, ids)
            item_embed = model(items, ids)
            sim_pos = F.cosine_similarity(user_embed, item_embed)
            users, items, ids = neg[0], neg[1], neg[2]
            users = users.to(device)
            items = items.to(device)
            user_embed = model(users, ids)
            item_embed = model(items, ids)
            sim_neg = F.cosine_similarity(user_embed, item_embed)
            target = torch.ones(sim_pos.numel()).to(device)
            loss = loss_fn(sim_pos, sim_neg, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'loss = {np.mean(losses)}')
        losses.clear()
        validate()
        print(f'epoch {epoch}: best_auc:{best_auc}\tbest_f1:{best_f1}\tbest_acc:{best_acc}\tbest_PatKs:{best_PatKs}')
        with open('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/saved_results/baseline.txt', 'a') as f:
            f.write(f'epoch {epoch}: best_auc:{best_auc}\tbest_f1:{best_f1}\tbest_acc:{best_acc}\tbest_PatKs:{best_PatKs}\n')
    print(f'final results on dev: best_auc:{best_auc}\nbest_f1:{best_f1}\nbest_acc:{best_acc}\nbest_PatKs:{best_PatKs}')
    with open('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/saved_results/baseline.txt', 'a') as f:
        f.write(f'final results on dev: best_auc:{best_auc}\nbest_f1:{best_f1}\nbest_acc:{best_acc}\nbest_PatKs:{best_PatKs}\n')

def evaluate():
    model.eval()
    print('evaluating')
    model.load_state_dict(torch.load("/home/xiajinxiong/workspace/bio/tanghaihong/temp_model/baseline_auc.pt",
                                     map_location="cuda:" + str(gpu_num)))
    print('evaluating auc score')
    preds, true = run_model(test_loader)
    auc = roc_auc_score(true, preds[0])

    model.load_state_dict(torch.load("/home/xiajinxiong/workspace/bio/tanghaihong/temp_model/baseline_f1.pt",
                                     map_location="cuda:" + str(gpu_num)))
    print('evaluating f1 score')
    preds, true = run_model(test_loader)
    f1 = f1_score(true, preds[0])

    model.load_state_dict(torch.load("/home/xiajinxiong/workspace/bio/tanghaihong/temp_model/baseline_acc.pt",
                                     map_location="cuda:" + str(gpu_num)))
    print('evaluating acc score')
    preds, true = run_model(test_loader)
    acc = accuracy_score(true, preds[0])

    print('evaluating PatK scores')
    PatKs = []
    for i in range(1, len(preds)):
        model.load_state_dict(torch.load(f"/home/xiajinxiong/workspace/bio/tanghaihong/temp_model/baseline_PatK{PatK_Ks[i - 1]}.pt",
                                         map_location="cuda:" + str(gpu_num)))
        preds, true = run_model(test_loader)
        PatKs.append(PatK_score(true, preds[i]))
    print(f'final results on test:\nauc:{auc}\nf1:{f1}\nacc:{acc}\nPatKs:{PatKs}')
    with open('/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/saved_results/baseline.txt', 'a') as f:
        f.write(f'final results on test:\nauc:{auc}\nf1:{f1}\nacc:{acc}\nPatKs:{PatKs}\n')

def main():
    load_data()
    optimizer_to(optimizer, device)
    train()
    evaluate()

if __name__ == "__main__":
    main()