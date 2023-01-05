import torch
from NLP.models.n_grams_embedder import NgramEmbedder, Trainer
from NLP.datasets.dataset import NgramsDs
from NLP.models.word_embeddings import WordEmbeds, EmbeddingsClassifier


def train_ngrams_embedder(language):

    for option in ["word_ngrams", "pos_ngrams", "dependency_ngrams"]:
        features_options = {
            "word_ngrams": False,
            "pos_ngrams": False,
            "dependency_ngrams": False,
            "domain_features": False,
            "doc_len": False,
        }

        features_options[option] = True

        dataset = NgramsDs(language, features_options)
        embed_size = dataset[0][0].shape[0]
        num_cls = len(torch.unique(torch.tensor([sample[-1] for sample in dataset])))

        model = NgramEmbedder(embed_size, num_cls)

        trainer = Trainer(
            model, dataset, num_epochs=200, lr=1e-3, batch_size=32, device="cuda:0"
        )

        model = trainer.train()

        torch.save(model.state_dict(), f"./saved_models/{language}_{option}.pt")


def train_lang_embeddings(language, use_char_embeddings=False):

    dataset = WordEmbeds(language)
    num_cls = len(torch.unique(torch.tensor([sample[-1] for sample in dataset])))
    model = EmbeddingsClassifier(
        word_vocab_size=len(dataset.word_vocab) + 2,
        char_vocab_size=len(dataset.char_vocab) + 2,
        num_classes=num_cls,
        word_max_seq_len=dataset[0][0].shape[-1],
        char_max_seq_len=dataset[0][1].shape[-1],
    )

    trainer = Trainer(
        model,
        dataset,
        num_epochs=100,
        lr=1e-2,
        batch_size=32,
        device="cuda:0",
        ds_is_ind=True,
    )

    model = trainer.train(use_char_embeddings)


def train_cross_language_NN(train_language, test_language):

    for option in ["pos_ngrams", "dependency_ngrams"]:

        features_options = {
            "word_ngrams": False,
            "pos_ngrams": False,
            "dependency_ngrams": False,
            "domain_features": False,
            "doc_len": False,
        }

        features_options[option] = True

        train_dataset = NgramsDs(train_language, features_options)
        test_dataset = NgramsDs(
            test_language, features_options, train_dataset.vectorizer
        )

        embed_size = train_dataset[0][0].shape[0]
        num_cls = len(
            torch.unique(torch.tensor([sample[1] for sample in train_dataset]))
        )

        model = NgramEmbedder(embed_size, num_cls)

        trainer = Trainer(
            model,
            train_dataset,
            num_epochs=200,
            lr=1e-3,
            batch_size=32,
            device="cuda:0",
            test_dataset=test_dataset,
        )

        model = trainer.train()

        torch.save(
            model.state_dict(),
            f"./saved_models/cross_{train_language}_{test_language}_{option}.pt",
        )

        f_score = trainer.eval_test_set()
        print("Test f1", f_score)


if __name__ == "__main__":

    # train_model = ["monolingual", "cross_lingual", "multi_lingual"]
    train_model = "ngrams_embedder"

    if train_model in ["monolingual", "multi_lingual"]:
        # language in ["czech", "german", "italian", "all"], "all" - multilingual
        language = "all"
        # train_lang_embeddings('czech')
        train_lang_embeddings(language, use_char_embeddings=False)

    elif train_model == "cross_lingual":

        train_lang = "german"

        test_lang = "italian"  # or czech
        train_cross_language_NN(train_lang, test_lang)

    elif train_model == "ngrams_embedder":
        train_ngrams_embedder('czech')


# german 0.6388853074773033
# italian  0.7950969390294895
# czech 0.629954909690814

# multi_langual - word embeddings - 0.6921900183887152
#                - word + char embeddings - 0.707827124628092
