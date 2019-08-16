import json
import random

from config import meta_file, vacab_file


def process_data():
    with open(meta_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    samples = []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            tokens = line.split()
            audiopath = 'data/BZNSYP/Wave/{}.wav'.format(tokens[0])
            text = lines[i+1].strip()
            for token in text:
                if token in ('P', 'I', 'Y'):
                    print(text)
                build_vocab(token)
            samples.append('{}|{}\n'.format(audiopath, text))

    valid_ids = random.sample(range(len(samples)), 100)
    train = []
    valid = []
    for id in range(len(samples)):
        sample = samples[id]
        if id in valid_ids:
            valid.append(sample)
        else:
            train.append(sample)

    # print(samples)
    with open('filelists/bznsyp_audio_text_train_filelist.txt', 'w', encoding='utf-8') as file:
        file.writelines(train)
    with open('filelists/bznsyp_audio_text_valid_filelist.txt', 'w', encoding='utf-8') as file:
        file.writelines(valid)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {}
    IVOCAB = {}

    process_data()

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB

    with open(vacab_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print('vocab_size: ' + str(len(data['VOCAB'])))
