import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


class NgramEmbedder(nn.Module):
    def __init__(self, embed_size, num_cls):

        super().__init__()

        self.embed_size = embed_size

        self.fc1 = nn.Linear(embed_size, 256)
        self.net = nn.Sequential(
            nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, num_cls)
        )

    def forward(self, x, y):

        logits = self.net(x)

        return logits


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        num_epochs,
        lr,
        batch_size,
        device,
        ds_is_ind=False,
        test_dataset=None,
    ):
        self.num_epochs = num_epochs
        self.lr = lr

        self.model = model.to(device)
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        # self.dataset = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=2)

        if test_dataset is not None:
            self.test_dataset = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )

        self.dataset = dataset
        self.device = device
        self.indices = ds_is_ind

    def eval_test_set(self):
        return self._compute_f1_score(self.model, self.test_dataset, average=None)

    def _compute_f1_score(
        self, model, dataloader, average="weighted", use_char_embeddings=False
    ):

        model = model.eval()
        all_preds = torch.empty(0).to(self.device)
        all_labels = torch.empty(0).to(self.device)

        with torch.no_grad():

            for data, char_data, labels in dataloader:

                if self.indices:
                    data = data.to(self.device).long()
                    char_data = char_data.to(self.device).long()

                else:
                    data = data.to(self.device).float()

                labels = labels.to(self.device)

                logits = model(
                    data, char_data if use_char_embeddings else None
                ).detach()

                preds = torch.argmax(logits, dim=-1)

                all_preds = torch.cat([all_preds, preds])
                all_labels = torch.cat([all_labels, labels])

        # print("F1 score", f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted'))

        return f1_score(
            all_labels.cpu().numpy(), all_preds.cpu().numpy(), average=average
        )

    def train(self, use_char_embeddings=False):
        splits = KFold(n_splits=10, shuffle=True, random_state=42)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

        folds_f1 = []

        for fold, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(self.dataset)))
        ):

            self.model.apply(weights_init)
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

            print("Fold", fold)

            train_sampler = Subset(self.dataset, train_idx)
            test_sampler = Subset(self.dataset, val_idx)
            train_loader = DataLoader(
                train_sampler, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(test_sampler, batch_size=self.batch_size)

            for epoch in tqdm(range(self.num_epochs)):

                self.model = self.model.train()

                epoch_loss = []

                for data, char_data, labels in train_loader:

                    if self.indices:
                        data = data.to(self.device).long()
                        char_data = char_data.to(self.device).long()

                    else:
                        data = data.to(self.device).float()

                    labels = labels.to(self.device)

                    logits = self.model(
                        data, char_data if use_char_embeddings else None
                    )

                    self.optimizer.zero_grad()

                    loss = self.criterion(logits, labels)

                    epoch_loss.append(loss.item())

                    loss.backward()
                    self.optimizer.step()

                mean_loss = sum(epoch_loss) / len(self.dataset)

                # self._compute_f1_score(self.model, train_loader)
                # print(f"Epoch {epoch + 1}  train_loss: {mean_loss}")

            folds_f1.append(
                self._compute_f1_score(
                    self.model,
                    test_loader,
                    average="weighted",
                    use_char_embeddings=use_char_embeddings,
                )
            )

            print(folds_f1[-1])

        print("Weighted f1", sum(folds_f1) / 10)

        import pdb; pdb.set_trace()
        return self.model
