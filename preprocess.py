import json
import nltk
from tqdm import tqdm
import pickle


def get_line_number(file_name):
    line_counter = 0
    with open(file_name, 'r') as f:
        for line in f:
            line_counter += 1
    return line_counter


def replace_special_tokens(sentence):
    sentence = sentence.replace('\\', '//')
    sentence = sentence.replace('{', '(')
    sentence = sentence.replace('}', ')')
    sentence = sentence.replace('^', '')
    return sentence


if __name__ == '__main__':
    random_seed = 123
    abstract_len_min, abstract_len_max = 50, 256
    title_len_min, title_len_max = 4, 20
    abstract_skip_words = [
        'withdrawn',
    ]
    title_skip_words = [
        'reply', 'Reply',
        'comment', 'Comment',
    ]
    target_category = 'cs.AI'

    data_fname = 'data/arxiv-metadata-oai-snapshot.json'
    line_number = get_line_number(data_fname)

    count = 0
    abstract_all = []
    title_all = []
    title_pos_all = []
    with open(data_fname, 'r') as f:
        for line in tqdm(f, total=line_number):
            paper = json.loads(line)

            if not any([cat == target_category for cat in paper['categories'].split()]):
                continue

            abstract = paper['abstract'].strip().replace('\n', ' ')
            title = paper['title'].strip().replace('\n  ', ' ')
            abstract = replace_special_tokens(abstract)
            title = replace_special_tokens(title)
            abstract_len = len(abstract.split())
            title_len = len(title.split())

            if abstract_len < abstract_len_min or abstract_len > abstract_len_max or \
                title_len < title_len_min or title_len > title_len_max or \
                any([w in abstract for w in abstract_skip_words]) or \
                any([w in title for w in title_skip_words]):
                continue

            title_tokens = nltk.word_tokenize(title.lower())
            title_pos = [pos for (word, pos) in nltk.pos_tag(title_tokens)]

            abstract_all.append(abstract)
            title_all.append(title)
            title_pos_all.append(title_pos)
            count += 1
            # if count == 100:
            #     break

    print(f"dataset size: {count}")

    with open('data/preprocessed.pkl', 'wb') as f:
        pickle.dump({
            'abstract': abstract_all,
            'title': title_all,
            'title_pos': title_pos_all,
        }, f)
