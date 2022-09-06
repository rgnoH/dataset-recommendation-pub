import os
import sys
import re
import requests
import time
from lxml import etree
from tqdm import tqdm as tq

from threading import Thread
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import RLock

PMID = set()    # GEO database papers' PMIDs
PMIDLock = RLock()

'''below: html processing'''

def isinGEO(pmid):
    PMIDLock.acquire()
    res = pmid in PMID
    PMIDLock.release()
    return res

def get_pmid_from_citation(citation):
    for elem in citation.iter(tag=etree.Element):
        if elem.tag == "pub-id" and elem.attrib.get("pub-id-type", None) == "pmid":
            return elem.text
    return None

def get_rid(citation):
    raw_rid = citation.attrib.get("id", None)
    if raw_rid == None:
        return None
    num = re.compile(r'.*?(\d+)$').findall(raw_rid)
    pre = re.compile(r'(.*?)\d+$').findall(raw_rid)
    if len(num) != 1 or len(pre) != 1:
        return None
    num, pre = str(int(num[0])), pre[0] # str(int()) 为了消除前导零
    return pre + num

def get_citation_dict(citation_list): # 提取出所有 citation 的 PMID，建立编号和 PMID 的映射
    rid2PMID = {}
    for citation in citation_list:
        rid = get_rid(citation) # 这个函数会消除编号中的前导零
        pmid = get_pmid_from_citation(citation)
        if rid != None:
            rid2PMID[rid] = pmid
    return rid2PMID

def iscitation(elem, rid2PMID):
    return elem.tag == "xref" and elem.attrib.get("rid", None) in rid2PMID.keys()

def fill_between(elem_l, elem_r, rid2PMID): # excluding l and r
    # print('hehe')
    res = ""
    pmid_l = elem_l.attrib.get("rid", None)
    pmid_r = elem_r.attrib.get("rid", None)
    if pmid_l == None or pmid_r == None:
        return res
    lnum = re.compile(r'.*?(\d+)').findall(pmid_l)
    lpre = re.compile(r'(.*?)\d+').findall(pmid_l)
    rnum = re.compile(r'.*?(\d+)').findall(pmid_r)
    rpre = re.compile(r'(.*?)\d+').findall(pmid_r)
    if len(lnum) != 1 or len(rnum) != 1 or lpre[0] != rpre[0]:
        return res
    lnum = int(lnum[0])
    rnum = int(rnum[0])
    pre = lpre[0]
    if lnum >= rnum - 1:
        return res
    # print(f'lnum: {lnum} rnum: {rnum} pre: {pre}')
    for i in range(lnum + 1, rnum):
        pmid = rid2PMID.get(pre + str(i), None)
        if pmid != None and isinGEO(pmid):  # 将 passage 里的 citation 换成 ##@@PMIDxxxx@@## 的形式
            res += '##@@PMID' + pmid + '@@##' + ', '
            # print(f'l: {lnum} r: {rnum} i: {i} pmid: {pmid}')
    return res

def check_passage_with_ref(passage, rid2PMID):
    for elem in passage.iter(tag=etree.Element):
        if iscitation(elem, rid2PMID):
            return True
    return False

def get_passages_with_ref(raw_passages, rid2PMID):
    # 由于 xref 可能是 supplicant files, figures 等，所以这里只提取出 rid 在 rid2PMID 中的 passage
    passages = []
    for passage in raw_passages:
        if check_passage_with_ref(passage, rid2PMID):
            passages.append(passage)
    return passages

# 这部分代码用来找引用多个文章时，中间的连接符，例如 3-5 之中的 -
# 最后找到了 - , â , â
# bridge_set = set()
# bridge_set_pmid = set()
# bridge_set_lock = RLock()

def dfs_Element(elem, rid2PMID):
    res = ""
    if elem.text != None:
        if iscitation(elem, rid2PMID):  # citation anchor
            pmid = rid2PMID.get(elem.attrib.get("rid", None), None)
            if pmid != None and isinGEO(pmid):   # 将 passage 里的 citation 换成 ##@@PMIDxxxx@@## 的形式
                res += '##@@PMID' + pmid  + '@@##'
        else:
            res += elem.text
    last_subelem = None
    for subelem in elem:
        if last_subelem != None \
                and iscitation(subelem, rid2PMID) and iscitation(last_subelem, rid2PMID) \
                and last_subelem.tail != None and last_subelem.tail.strip() in ('â', '-', 'â'):
            res += fill_between(last_subelem, subelem, rid2PMID)  # 填充 x-y 的形式
            # 这部分代码用来找引用多个文章时，中间的连接符，例如 3-5 之中的 -
            # if last_subelem.tail != None and len(last_subelem.tail) < 5:
            #     with bridge_set_lock:
            #         bridge_set.add(last_subelem.tail)
        last_subelem = subelem
        res += dfs_Element(subelem, rid2PMID)
    if elem.tail != None:
        res += elem.tail
    return res

def process_passage(passage, rid2PMID):
    return dfs_Element(passage, rid2PMID)

def process_xml_tree(pmid, tree):
    raw_passages = tree.xpath("//p[descendant::xref]")
    citation_list = tree.xpath("//ref-list//*[@id]")
    rid2PMID = get_citation_dict(citation_list)
    passages = get_passages_with_ref(raw_passages, rid2PMID)
    if len(passages) == 0 or len(citation_list) == 0:
        return
    res = ""
    for passage in passages:
        processed_passage = process_passage(passage, rid2PMID).strip()
        check_pmid = re.compile(r'##@@PMID(\d+)@@##').search(processed_passage)
        if check_pmid != None:
            res += processed_passage + '\n----------\n'
    if res != "":
        f = open('/home/xiajinxiong/workspace/bio/tanghaihong/data/recrawl/' + pmid + '.txt', 'w', encoding='utf-8')
        f.write(res)
        f.close()

'''below: crawler'''

all_task = []
all_task_index = 0
all_task_lock = RLock()
print_lock = RLock()
process_next = 1000
process_interval = 1000
process_start_time = time.time()
process_cur_time = time.time()
process_last_time = time.time()

def get_next_task():
    global all_task, all_task_index, process_interval, process_next
    global process_cur_time, process_last_time, process_start_time
    all_task_lock.acquire()
    if all_task_index >= len(all_task):
        all_task_lock.release()
        return None
    res = all_task[all_task_index]
    if process_next == all_task_index:
        process_next += process_interval
        process_cur_time = time.time()
        eta = (process_cur_time - process_start_time) / (all_task_index + 1) * (len(all_task) - all_task_index - 1)
        with print_lock:
            print(f'[INFO] {all_task_index} tasks finished | '
                  f'{process_cur_time - process_last_time:.2f} secs for this {process_interval} | '
                  f'{process_cur_time - process_start_time:.2f} secs in total | '
                  f'eta: {eta // 3600:.0f}h\t{eta % 3600 // 60:.0f}min\t{eta % 60:.2f}sec')
        process_last_time = process_cur_time
    all_task_index += 1
    all_task_lock.release()
    return res

def crawl(cid):
    while True:
        # time.sleep(0.5)
        tmp = get_next_task()
        if tmp == None:
            break
        pmid, pmcid = tmp
        # print(f'pmcid: {pmcid}')
        url = 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:' + pmcid + '&metadataPrefix=pmc'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
        req = requests.get(url, headers=headers)
        if req.status_code == 200:
            tree = etree.HTML(req.text.encode('utf-8'))
            process_xml_tree(pmid, tree)
        else:
            with print_lock:
                print(f'status code {req.status_code} when accessing {pmcid}')
    # with print_lock:
    #     print(f'[INFO] thread {cid} completed')

def init():
    global PMID, all_task
    file_names = os.listdir('/home/xiajinxiong/workspace/bio/data/pmid2xml')
    PMID = set(file_names)  # 只考虑 pmid2xml，即 GEO database paper的 PMID
    with open('/home/xiajinxiong/workspace/bio/tanghaihong/data/pmid2pmc_crawl.txt', 'r') as f:
        for line in f.readlines():
            if len(line.strip().split()) == 2:
                pmid, pmcid = line.strip().split()
                all_task.append((pmid, pmcid[3:]))
    # all_task = all_task[:10000]
    print('[INFO] initialization completed')

def main():
    init()
    with ThreadPoolExecutor(max_workers=200) as pool:
        results = pool.map(crawl, range(200))
        tmp = list(results)
        print('[INFO] crawling completed')

if __name__ == "__main__":
    main()