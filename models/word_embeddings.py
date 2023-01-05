from NLP.datasets.dataset import prepare_dataset_word_embeds
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WordEmbeds(Dataset):
    def __init__(self, language):

        super().__init__()

        self.word_dataset, self.labels, self.word_vocab = prepare_dataset_word_embeds(
            language
        )
        self.char_dataset, _, self.char_vocab = prepare_dataset_word_embeds(
            language, is_char=True, seq_max_len=2000
        )

        assert len(self.word_dataset) == len(self.labels)
        assert len(self.char_dataset) == len(self.labels)

    def __getitem__(self, index):
        return self.word_dataset[index], self.char_dataset[index], self.labels[index]

    def __len__(self):
        return len(self.word_dataset)


class EmbeddingsClassifier(nn.Module):
    def __init__(
        self,
        word_vocab_size,
        char_vocab_size,
        num_classes,
        word_max_seq_len=400,
        char_max_seq_len=400,
        word_embedding_size=400,
        char_embedding_size=16,
    ):

        super().__init__()
        self.word_embedding = nn.Embedding(
            word_vocab_size, word_embedding_size, padding_idx=0
        )
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embedding_size, padding_idx=0
        )

        self.comb_linear = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(
                word_max_seq_len * word_embedding_size
                + char_max_seq_len * char_embedding_size,
                num_classes,
            ),
        )

        self.num_classes = num_classes

        self.word_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(word_max_seq_len * word_embedding_size, num_classes),
        )

    def forward(self, words, chars):

        words = self.word_embedding(words)

        if chars is not None:
            chars = self.char_embedding(chars)
            out = torch.cat(
                [words.view(words.shape[0], -1), chars.view(words.shape[0], -1)], dim=-1
            )
            out = self.comb_linear(out)

        else:
            out = self.word_linear(words)

        return out
