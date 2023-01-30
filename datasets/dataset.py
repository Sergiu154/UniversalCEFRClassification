import numpy as np
import os
import re
import string
from tqdm import tqdm
import spacy_udpipe
from sklearn.feature_extraction.text import CountVectorizer
from NLP.models.n_grams_embedder import NgramEmbedder
import torch
import pickle
import os
import string
from transformers import AutoModel, AutoTokenizer



DS_CACHE = {}


def load_text_and_cefr_score(text_file_path, text_score_path):

    with open(text_file_path) as fin:
        text = " ".join([x.strip() for x in fin.readlines()])

    with open(text_score_path) as fin:
        score = fin.readlines()[12].strip().split(" ")[-1]

    return text, score


def create_ds(language):

    if os.path.exists(f"./datasets/pickled/{language}_ds.pickle"):

        with open(f"./datasets/pickled/{language}_ds.pickle", "rb") as handle:
            ds = pickle.load(handle)

        return ds

    imp_to_lang = {"it": "italian", "de": "german", "cs": "czech"}

    language_map = {
        "italian": ["it"],
        "german": ["de"],
        "czech": ["cs"],
        "all": ["de", "it", "cs"],
    }
    ds = {}

    for lm in language_map[language]:

        spacy_udpipe.download(lm)
        tokenizer = spacy_udpipe.load(lm)

        ds_text_path = f"./merlin-text-v1.1/plain/{imp_to_lang[lm]}"
        ds_rating_path = f"./merlin-text-v1.1/meta_ltext/{imp_to_lang[lm]}"

        if lm == "cs":

            cs_parsed_files = f"datasets/CZ-Parsed/"
            doc_mapper = {
                f"{x.split('_')[0]}.txt": parse_czech_file(
                    os.path.join(cs_parsed_files, x)
                )
                for x in os.listdir(cs_parsed_files)
            }

        for file in tqdm(os.listdir(ds_text_path)):

            text, score = load_text_and_cefr_score(
                os.path.join(ds_text_path, file), os.path.join(ds_rating_path, file)
            )

            if len(score) == 0 or (lm == "cs" and file not in doc_mapper):
                print(file)
                continue

            sample_data = {
                "file_name": file,
                "plain_text": text,
                "tokenized_docs": tokenizer(re.sub(r"[^\w\s]", "", text))
                if lm != "cs"
                else doc_mapper[file],
            }

            sample_data["word"] = create_encodings(
                sample_data["tokenized_docs"], type="word"
            )
            sample_data["dep"] = create_encodings(
                sample_data["tokenized_docs"], type="dependency"
            )
            sample_data["pos"] = create_encodings(
                sample_data["tokenized_docs"], type="pos"
            )
            sample_data["score"] = score
            sample_data["language"] = lm

            if score not in ds:
                ds[score] = [sample_data]

            else:
                ds[score].append(sample_data)

    keys = list(ds.keys())
    for score in keys:

        if len(ds[score]) <= 10:
            del ds[score]

    with open(f"./datasets/pickled/{language}_ds.pickle", "wb") as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ds


def create_domain_features(docs):

    domain_features = np.zeros((len(docs), 9))

    for index, doc in enumerate(docs):

        doc = doc["tokenized_docs"]
        total = len(doc)
        total_dist_words = len(list(map(lambda x: x.lemma_, doc)))
        nouns = list(
            map(lambda x: x.lemma_, filter(lambda x: x.pos_ in ["NOUN", "PROPN"], doc))
        )
        adj = list(map(lambda x: x.lemma_, filter(lambda x: x.pos_ == "ADJ", doc)))
        adv = list(map(lambda x: x.lemma_, filter(lambda x: x.pos_ == "ADV", doc)))
        interj = list(map(lambda x: x.lemma_, filter(lambda x: x.pos_ == "INTJ", doc)))
        verbs = list(map(lambda x: x.lemma_, filter(lambda x: x.pos_ == "VERB", doc)))

        total_lex = float(
            len(nouns) + len(adj) + len(adv) + len(interj) + len(verbs)
        )  # Total Lexical words i.e., tokens
        distinct_lex = float(
            len(set(nouns))
            + len(set(adj))
            + len(set(adv))
            + len(set(interj))
            + len(set(verbs))
        )  # Distinct Lexical words i.e., types

        result = np.array(
            [
                total,
                round(total_dist_words / total, 2),
                round(total_lex / total, 2),
                round(distinct_lex / total_lex, 2),
                round(len(set(verbs)) / total_lex, 2),
                round(len(set(nouns)) / total_lex, 2),
                round(len(set(adj)) / total_lex, 2),
                round(len(set(adv)) / total_lex, 2),
                round((len(set(interj)) + len(set(adv))) / total_lex, 2),
            ]
        )

        domain_features[index] = result

    return domain_features


def create_encodings(doc, type="word"):

    if type == "word":

        return " ".join([x.lemma_ for x in doc])

    elif type == "dependency":

        return " ".join(["_".join([x.dep_, x.pos_, x.head.pos_]) for x in doc])

    elif type == "pos":
        return " ".join([x.pos_ for x in doc])


def create_ngrams(docs, vectorizer, n_gram_type="word"):

    raw_text = [doc[n_gram_type] for doc in docs]

    if vectorizer is None:
        vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None,
            ngram_range=(1, 5),
            min_df=10,
        )

        feature_vectors = vectorizer.fit_transform(raw_text)

    else:
        feature_vectors = vectorizer.transform(raw_text)

    return feature_vectors.toarray(), vectorizer


def get_bert_embeddings(docs):

    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    model = AutoModel.from_pretrained("bert-base-german-cased").to('cuda:0')
    features = torch.empty((0, 768)).to('cuda:0')

    with torch.no_grad():

        for doc in docs:

            out = tokenizer(doc["word"])
            input_ids = torch.tensor(out['input_ids']).to('cuda:0').unsqueeze(0)
            att_mask = torch.tensor(out['attention_mask']).to('cuda:0').unsqueeze(0)
            hidden_states, pooled_output = model(input_ids= input_ids, attention_mask = att_mask, return_dict=False)

            # features = torch.cat([features, pooled_output], dim =0)
            features = torch.cat([features,hidden_states.mean(dim=1)], dim =0)

    return features.cpu().numpy()

def compress_features(language, feat_type, features, labels):

    imp_to_lang = {"it": "italian", "de": "german", "cs": "czech"}

    lang_map = {v: k for k, v in imp_to_lang.items()}

    model = NgramEmbedder(
        features.shape[1], len(torch.unique(torch.tensor(labels)))
    ).to("cuda:0")
    model.load_state_dict(
        torch.load(f"./saved_models/{lang_map[language]}_{feat_type}.pt")
    )
    th_features = torch.tensor(features).to("cuda:0").float()

    with torch.no_grad():

        logits = model(th_features, None)
        probs_feat = torch.nn.functional.softmax(logits, dim=-1)

    return probs_feat.cpu().numpy()


def create_feature_vector2(
    docs,
    language,
    vectorizer,
    class_map=None,
    word_ngrams=True,
    pos_ngrams=True,
    dependency_ngrams=True,
    domain_features=True,
    doc_len=False,
    language_flag=False,
    bert_embeddings=False
):

    features = np.empty((len(docs), 0))

    labels = []

    for doc in docs:
        labels.append(doc["score"])

    unique_labels = np.unique(labels)

    label_to_indx = {ul: index for index, ul in enumerate(unique_labels)}
    label_to_indx = label_to_indx if not class_map else class_map

    labels = [label_to_indx[label] for label in labels]
    labels = np.array(labels)

    if bert_embeddings:
        features = get_bert_embeddings(docs)
        vectorizer = None

    if word_ngrams:

        word_feat, vectorizer = create_ngrams(docs, vectorizer, "word")

        if domain_features:
            word_feat = compress_features(language, "word_ngrams", word_feat, labels)

        features = np.concatenate([features, word_feat], axis=-1)

    if dependency_ngrams:

        dep_feat, vectorizer = create_ngrams(docs, vectorizer, "dep")

        if domain_features:
            word_feat = compress_features(
                language, "dependency_ngrams", dep_feat, labels
            )

        features = np.concatenate([features, dep_feat], axis=-1)

    if pos_ngrams:

        pos_feat, vectorizer = create_ngrams(docs, vectorizer, "pos")
        if domain_features:
            word_feat = compress_features(language, "pos_ngrams", pos_feat, labels)

        features = np.concatenate([features, pos_feat], axis=-1)

    if domain_features or doc_len:

        dom_features = create_domain_features(docs)

        features = np.concatenate([features, dom_features], axis=-1)

        if doc_len:
            features = features[:, [0]]

    if language_flag:

        lang_to_idx = {"de": 0, "it": 1, "cs": 2}
        lang_flags = np.zeros((len(docs), 3))
        for index, doc in enumerate(docs):
            lang_flags[index, lang_to_idx[doc["language"]]] = 1.0

        features = np.concatenate([features, lang_flags], axis=-1)

    return features, labels, vectorizer, label_to_indx


def labels_to_idx(labels):
    unique_labels = np.unique(labels)

    label_to_indx = {ul: index for index, ul in enumerate(unique_labels)}
    labels = [label_to_indx[label] for label in labels]
    labels = np.array(labels)

    return labels


def create_feature_vector(
    dataset,
    word_ngrams=True,
    pos_ngrams=True,
    dependency_ngrams=True,
    domain_features=True,
    doc_len=False,
):

    docs = []
    labels = []

    for label, documents in dataset.items():

        for doc in documents:
            docs.append(doc["tokenized_docs"])
            labels.append(label)

    unique_labels = np.unique(labels)

    label_to_indx = {ul: index for index, ul in enumerate(unique_labels)}
    labels = [label_to_indx[label] for label in labels]
    labels = np.array(labels)

    features = np.empty((len(docs), 0))
    if word_ngrams:

        for N in range(1, 6):

            word_features = create_ngrams(docs, N, n_gram_type="word")

            features = np.concatenate([features, word_features], axis=-1)

    if pos_ngrams:

        for N in range(1, 6):

            word_features = create_ngrams(docs, N, n_gram_type="pos")
            features = np.concatenate([features, word_features], axis=-1)

    if dependency_ngrams:

        for N in range(1, 6):

            word_features = create_ngrams(docs, N, n_gram_type="dependency")
            features = np.concatenate([features, word_features], axis=-1)

    if domain_features or doc_len:
        dom_features = create_domain_features(docs)

        features = np.concatenate([features, dom_features], axis=-1)

        if doc_len:
            features = features[:, [0]]

    return features, labels


def create_dataset_and_extract_features(
    language, features_options, vectorizer, class_map
):

    if language not in DS_CACHE:
        dataset = create_ds(language)
        DS_CACHE[language] = dataset
    else:
        dataset = DS_CACHE[language]

    docs = [x for value in dataset.values() for x in value]
    return create_feature_vector2(
        docs, language, vectorizer, class_map, **features_options
    )


from torch.utils.data import Dataset


class NgramsDs(Dataset):
    def __init__(self, language, features_options, vectorizer=None):

        super().__init__()

        (
            self.dataset,
            self.labels,
            self.vectorizer,
            _,
        ) = create_dataset_and_extract_features(language, features_options, vectorizer, class_map = None)

        assert len(self.dataset) == len(self.labels)

    def __getitem__(self, index):
        return self.dataset[index], torch.tensor(0), self.labels[index]

    def __len__(self):

        return len(self.dataset)


def create_vocab(language, is_char=False, min_freq=15):

    if language not in DS_CACHE:
        dataset = create_ds(language)
        DS_CACHE[language] = dataset
    else:

        dataset = DS_CACHE[language]

    docs = [(label, x) for label, value in dataset.items() for x in value]

    word_count = {}

    for label, doc in docs:

        for token in doc["tokenized_docs"]:

            if is_char:

                for char in token.lemma_:
                    word_count[char] = word_count.get(char, 0) + 1
            else:
                word_count[token.lemma_] = word_count.get(token.lemma_, 0) + 1

    words = list(word_count.keys())

    for word in words:
        if word_count[word] < min_freq:
            del word_count[word]

    vocab = {word: index + 1 for index, word in enumerate(list(word_count.keys()))}

    return vocab, docs


def prepare_dataset_word_embeds(language, is_char=False, min_freq=15, seq_max_len=400):
    vocab, docs = create_vocab(language, is_char, min_freq)

    X = []

    pad_index = 0
    oov_index = 1
    labels = []

    for label, doc in docs:
        sample = []

        labels.append(label)
        for token in doc["tokenized_docs"]:

            if is_char:

                for char in token.lemma_:
                    sample.append(vocab.get(char, oov_index))

            else:
                sample.append(vocab.get(token.lemma_, oov_index))

        X.append(torch.tensor(sample))

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)[:, :seq_max_len]

    labels = labels_to_idx(labels)
    labels = torch.tensor(labels)

    return X, labels, vocab


class Token:
    def __init__(self, word, lemma, pos, dep):

        self.word_ = word
        self.lemma_ = lemma
        self.pos_ = pos
        self.dep_ = dep


class Doc:
    def __init__(self, plain_text, lemmas, POS, DEP_POS, dependencies):

        self.plain_text = plain_text
        self.lemmas = lemmas
        self.DEP_POS = DEP_POS
        self.POS = POS

        self.dependencies = dependencies

        self.tokens = []

        for word, lemma, pos, dep, head in zip(
            self.plain_text, self.lemmas, self.POS, self.DEP_POS, self.dependencies
        ):
            token = Token(word, lemma, pos, dep)
            token.head = Token(head[0], head[0], head[-1], head[1])

            self.tokens.append(token)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):

        return self.tokens[index]

    def __repr__(self):

        return " ".join(self.plain_text)


def parse_czech_file(file_path):

    print(file_path)
    plain_text = ""
    lemmas = ""
    POS = ""
    DEP_POS = ""
    last_index = -1
    current_line = []

    lemma_to_head = []
    skip_line = False

    with open(file_path, "r") as fin:

        for line in fin.readlines()[2:]:

            if len(line.strip()) == 0:

                if skip_line:
                    skip_line = False
                    continue
                # last_index = len(plain_text.strip()) - 1
                for lemma, head_idx, curr_dep, curr_dep_pos in current_line:

                    if head_idx - 1 == -1:
                        dep_word = "root"
                        dep_type = curr_dep
                        dep_pos = curr_dep_pos

                    else:
                        dep_word = current_line[head_idx - 1][0]

                        dep_type, dep_pos = current_line[head_idx - 1][-2:]

                    lemma_to_head.append((dep_word, dep_type, dep_pos))

                current_line = []
                continue

            split_line = line.strip().split("\t")

            if line.startswith("#"):
                if "text" in line:
                    pass
                else:
                    continue

            else:

                try:
                    word, lemma, pos = split_line[1:4]

                    dep_pos = split_line[-3]

                    current_line.append((word, int(split_line[-4]), dep_pos, pos))

                    plain_text += word + " "
                    last_index += 1

                except Exception as e:
                    print("warning", e, file_path)
                    skip_line = True
                    continue

                lemmas += " " + lemma
                POS += " " + pos
                DEP_POS += " " + dep_pos

    plain_text = plain_text.strip()
    lemmas = lemmas.strip()
    POS = POS.strip()
    DEP_POS = DEP_POS.strip()

    inds = {
        index: True
        for index, word in enumerate(plain_text.split())
        if word not in string.punctuation
    }

    plain_text = [x for index, x in enumerate(plain_text.split()) if index in inds]
    lemmas = [x for index, x in enumerate(lemmas.split()) if index in inds]
    POS = [x for index, x in enumerate(POS.split()) if index in inds]
    DEP_POS = [x for index, x in enumerate(DEP_POS.split()) if index in inds]
    lemma_to_head = [x for index, x in enumerate(lemma_to_head) if index in inds]

    doc = Doc(plain_text, lemmas, POS, DEP_POS, lemma_to_head)

    return doc
