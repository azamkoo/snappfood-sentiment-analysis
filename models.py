import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, device):
        super(RNNClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx).to(device)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True).to(device)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(out).squeeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, device):
        super(LSTMClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx).to(device)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True).to(device)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(out).squeeze(1)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, device):
        super(GRUClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx).to(device)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True).to(device)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(out).squeeze(1)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=1, pad_idx=0, device='cpu',
                 num_filters=100, filter_sizes=[3,4,5], dropout=0.5):
        super(TextCNNClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx).to(device)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ]).to(device)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * num_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # because binary classification
        ).to(device)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim) for Conv2d

        # Apply convolution + ReLU + max pooling
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(embedded)).squeeze(3)  # remove last dim -> (batch_size, num_filters, seq_len-filter_size+1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outs.append(c)

        cat = torch.cat(conv_outs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        out = self.fc(cat)
        return out.squeeze(1)
    
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, device):
        super(BiLSTMClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(out).squeeze(1)