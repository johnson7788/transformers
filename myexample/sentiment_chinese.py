import torch

import random
import numpy as np
from torchtext import data
from torchtext import datasets
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import time


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

class BERTGRU(nn.Module):
    def __int__(self, model, hidden_dim=512, ouput_dim=1024, n_layers=2,
                bidirectional=True, dropout=0.5):
        super().__init__()
        self.model = model
        embedding_dim =self.model.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                          batch_first=True, dropout=0 if n_layers<2 else dropout)
        self.out = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, ouput_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        #input [batch size, len]
        with torch.no_grad():
            embedded = self.bert(input)[0]
        # embedded = [batch size, sent len, emb dim]

        #hidden = [n layers * n directions, batch size, emb dim]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


class DataHelper(object):
    def __init__(self, tokenizer, max_input_length=512, batch_size=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.batch_size = batch_size

    def get_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        init_token_idx = self.tokenizer.cls_token_id
        eos_token_idx = self.tokenizer.sep_token_id
        pad_token_idx = self.tokenizer.pad_token_id
        unk_token_idx = self.tokenizer.unk_token_id

        def tokenize_and_cut(sentence):
            tokens = self.tokenizer.tokenize(sentence)
            tokens = tokens[:self.max_input_length - 2]
            return tokens

        TEXT = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=self.tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx)

        LABEL = data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state=random.seed(SEED))
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.batch_size,
            device=device)
        return train_iterator, valid_iterator,test_iterator



class Network(object):
    def __init__(self, model='bert-base-chinese'):
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)

    #前向网络
    def interface(self, train=True):
        for name, param in self.model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        if train:
            model.train()
            for batch in data:
                self.optimizer.zero_grad()

                predictions = model(batch.text).squeeze(1)

                loss = loss(predictions, batch.label)

                acc = binary_accuracy(predictions, batch.label)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def optimizer(self):
        optim.Adam(self.model.parameters())
    def losses(self, predictions, label):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.BCEWithLogitsLoss()
        model = self.model.to(device)
        criterion = criterion.to(device)
        loss = criterion(predictions, label)

    def metrics(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def save(self, save_name):
        torch.save(self.model.state_dict(), save_name)



def train(model, data, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in data:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, data, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in data:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    model = Network()
    train_iterator, valid_iterator, test_iterator = DataHelper(tokenizer=model.tokenizer, max_input_length=512, batch_size=128)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model=model, data=train_iterator)
        valid_loss, valid_acc = evaluate(model=model, data=valid_iterator)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss or epoch % 3 == 0:
            best_valid_loss = valid_loss
            model.save(save_name='sentiment.bin')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')