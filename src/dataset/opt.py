import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import re
from tqdm import tqdm as tq
from os import listdir
from os.path import isfile, join


def load_model_tokenizer(model_name: str = '/home/xiajinxiong/workspace/huggingface/opt-1.3b'):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer


@torch.no_grad()
def generate_text(input_text: str, nlg_model: AutoModelForCausalLM, tokenizer, device: torch.device, nlp):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    generated_ids = nlg_model.generate(input_ids, max_new_tokens=100,
                                       num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generated_text = generated_text[len(input_text):].strip()
    generated_sentences = [i.text for i in nlp(generated_text).sents]
    target_text = ' '.join(generated_sentences)
    return target_text


def inference():
    nlp = spacy.load('en_core_web_sm')
    gpu_num = 0
    device = torch.device("cuda:" + str(gpu_num))
    model_name = "/home/xiajinxiong/workspace/huggingface/opt-1.3b"
    print("Loading model", model_name)
    model, tokenizer = load_model_tokenizer(model_name)
    model.to(device)
    print("Done")
    mypath = "/home/xiajinxiong/workspace/bio/tanghaihong/data/recrawl_processed"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    mark_pattern = re.compile("##@@PMID\d+@@##")
    replace_pattern = re.compile(' xia jinxiong ')
    prompt_file = open("/home/xiajinxiong/workspace/bio/data/prompt/prompt1_example.txt", "a")
    pair_num = 0
    for file_index, file in enumerate(tq(files)):
        items = re.split('[_.]', file)
        dataset_pmid = items[1]
        pmid = items[0]
        dataset_mark = '##@@PMID' + dataset_pmid + '@@##'
        # print(dataset_mark)
        full_file_name = join(mypath, file)
        with open(full_file_name, "r") as f:
            lines = f.readlines()
            f.close()
        for line in lines:
            if line[:10] == '----------':
                continue
            line = re.sub(dataset_mark, ' xia jinxiong ', line)
            line = re.sub(mark_pattern, '', line)
            sentences = [i.text for i in nlp(line).sents]
            index = -1
            context = ""
            for i, sentence in enumerate(sentences):
                # print(sentence)
                # print(dataset_mark)
                find = sentence.find('')
                if find != -1:
                    index = i
                    context = re.sub(replace_pattern, '', sentence)
                    context = context.strip()
                    # sentences[i] = re.sub(mark_pattern, '', sentence)
                    break
            if index == -1:
                continue
            if len(context.split()) < 10:
                continue
            input_text = "Dataset description: " + context + " What is the dataset used for?"
            output_text = generate_text(input_text, model, tokenizer, device, nlp)
            prompt_file.write("Input text=" + input_text)
            prompt_file.write("\n\n")
            prompt_file.write("Output text=" + output_text)
            prompt_file.write("\n\n")
            pair_num += 1
            if pair_num == 100:
                prompt_file.close()
                return


if __name__ == '__main__':
    inference()
