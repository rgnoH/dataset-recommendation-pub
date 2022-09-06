import os
import re
import csv
import random
from tqdm import tqdm as tq

# todo: what if the question of a dataset is the same as the other?

abstract_path = '/home/xiajinxiong/workspace/bio/tanghaihong/data/recrawl_abstract/'
summary_path = '/home/xiajinxiong/workspace/bio/data/gse2summary/'
question_path = '/home/xiajinxiong/workspace/bio/data/prompt/filtered_data.csv'
PMID2GSE_path = '/home/xiajinxiong/workspace/bio/tanghaihong/data/full_extracted_merged_series.csv'
dest_path = '/home/xiajinxiong/workspace/bio/tanghaihong/data/dataset/sample/'
all_data = []

dict_pmid2gse = {}
f = open(PMID2GSE_path, 'r', encoding='utf-8', newline='')
f_reader = csv.reader(f)
for row in f_reader:
    for pmid in row[8].split(';'):
        dict_pmid2gse[pmid] = row[0]
f.close()

# abstracts:    absPMID.txt => PMID
# summarys:     GSExxxx
# questions:    in csv, row[1] = PMIDdataset_PMIDcitation, row[2] = question

abstracts = set([file[3:-4] for file in os.listdir(abstract_path)])
# abstracts = set(os.listdir(abstract_path))
summarys = set(os.listdir(summary_path))
filtered = open(question_path, 'r', encoding='utf-8', newline='')
f_reader = csv.reader(filtered)
filtered_list = []
for row in f_reader:
    filtered_list.append((row[1], row[3]))
filtered.close()

# print(f'questions: {len(filtered_list)}')
# print(filtered_list[0])
# print(filtered_list[1])

# (question, abstract, summary, GSE, id)
# expected usage:
# user = question
# item = abstract [SEP] summary

for i, tup in tq(enumerate(filtered_list[1:])):
    file, question = tup[0], tup[1]
    pmid_dataset, pmid_citation = file.split('_')
    gse = dict_pmid2gse.get(pmid_dataset, None)
    if gse == None or gse not in summarys or pmid_dataset not in abstracts:
        continue
    f = open(abstract_path + 'abs' + pmid_dataset + '.txt', 'r', encoding='utf-8')
    abstract = f.read().strip()
    f.close()
    f = open(summary_path + gse, 'r', encoding='utf-8')
    summary = f.read().strip()
    f.close()
    all_data.append([question, abstract, summary, gse, i])

f = open(dest_path + 'alldata_filtered.csv', 'w', encoding='utf-8')
writer = csv.writer(f)
writer.writerows(all_data)
f.close()