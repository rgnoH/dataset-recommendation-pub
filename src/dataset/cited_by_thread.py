import csv
import os
import sys
import re
import requests
import time
from lxml import etree

from threading import Thread
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import RLock

'''
-----format-----
PMID num
PMID(cited_by)1 PMID(cited_by)2 ... PMID(cited_by)num
'''

dir_path = '../data/GSElists'
read_file = dir_path + "/GSE_PMIDs.txt"

with open(read_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))

total_set = set()
gse_set = set()
urls = []

for line in lines:
    line = line.strip()
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id={line}'
    urls.append(url)
    gse_set.add(line)

sz = 25000
url_slices = [urls[i * sz : min((i + 1) * sz, len(urls))] for i in range(0, (len(urls) + sz - 1) // sz)]

link_cnt = []
cnt = 0
mark = 100
start_time = time.time()
last_time = start_time
var_lock = RLock()

def print_link_info():
    avg_links = sum(link_cnt) / len(link_cnt)
    max_links = max(link_cnt)
    print(f'[INFO] average link: {avg_links}')
    print(f'[INFO] max link: {max_links}')
    print(f'[INFO] total link: {sum(link_cnt)}')
    print(f'[INFO] total link (unique): {len(total_set)}')

print_lock = RLock()
not_req = open(dir_path + "/not_req_urls.txt", "w", encoding='utf-8') # record urls failed to request
req_err = open(dir_path + "/req_err_urls.txt", "w", encoding='utf-8') # record urls whose status_code is not equal to 200
xml_err = open(dir_path + "/xml_err_urls.txt", "w", encoding='utf-8') # record urls whose xml format is wrong
zero_link = open(dir_path + "/zero_link_urls.txt", "w", encoding='utf-8') # record urls which is not cited by any other one

def crawl(cid):
    f = open(dir_path + "/cited_by" + str(cid) + ".txt", "w", encoding='utf-8')
    
    with var_lock:
        url_slice = url_slices[cid]
        print(f'{cid} url_slice: {len(url_slice)}')

    print(f'[INFO] start crawling cited_by{cid}.txt...')
    # for url in url_slice:
    for i in range(0, len(url_slice)):
        url = url_slice[i]
        time.sleep(1)
        try:
            res = requests.get(url)
        except:
            with print_lock:
                print(f'[INFO][ERROR] request failed when trying to get {url}')
                not_req.write(url + '\n')
            continue

        if(res.status_code != 200):
            with print_lock:
                print(f'[INFO][ERROR] status_code {res.status_code} when trying to get {url}')
                req_err.write(str(res.status_code) + ' ' + url + '\n')
            continue
   
        tree = etree.fromstring(res.text.encode('utf-8'))
        pmid = tree.xpath('//IdList/Id/text()')
        links = tree.xpath('//LinkSetDb/Link/Id/text()')
   
        if len(pmid) != 1:
            with print_lock:
                print(f'[INFO][ERROR] pmid = {len(pmid)} for {url}')
                xml_err.write(str(len(pmid)) + ' ' + url + '\n')
            continue
        pmid = pmid[0]
        
        if len(links) != 0:
            f.write(pmid + ' ' + str(len(links)) + '\n')
            f.write(' '.join(links))
            f.write('\n')
        else:
            with print_lock:
                zero_link.write(pmid + '\n')
        
        # other processing
        with var_lock:
            global cnt
            global mark
            global cur_time
            global last_time
            global start_time
            link_cnt.append(len(links))
            for link in links:
                total_set.add(link)
            cnt += 1
            if cnt == mark:
                mark += 100
                cur_time = time.time()
                with print_lock:
                    print(f'{cnt} [INFO][TIME] urls have been processed...')
                    print(f'{cnt} [INFO][TIME] total time: {cur_time - start_time} secs')
                    print(f'{cnt} [INFO][TIME] these 100 : {cur_time - last_time} secs')
                    print_link_info()
                last_time = cur_time

    print(f'[INFO] cited_by{cid}.txt completed\n')

    f.close()

def write_all_citations():
    # write all links in total_set in order to get full text
    with open(dir_path + "/citation_PMIDs.txt", "w", encoding='utf-8') as f:
        for link in total_set:
            f.write(link + '\n')

    diff_set = total_set - gse_set
    print(f'[INFO] diff set size: {len(diff_set)}')
    with open(dir_path + "/diff_citation_PMIDs.txt", "w", encoding='utf-8') as f:
        for link in diff_set:
            f.write(link + '\n')

def main():
    print(str((len(urls) + sz - 1) // sz))
    with ThreadPoolExecutor(max_workers=(len(urls) + sz - 1) // sz) as pool:
        results = pool.map(crawl, range(0, len(url_slices)))
        print(list(results))
    # time.sleep(20)
    print_link_info()
    write_all_citations()

    zero_link.close()
    xml_err.close()
    req_err.close()
    not_req.close()

if __name__ == '__main__':
    main()