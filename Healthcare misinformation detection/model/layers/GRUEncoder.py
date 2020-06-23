import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUEncoder, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # bigru encoder
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return nn.Parameter(torch.zeros(2, 1, self.hidden_dim))

    def forward(self, text_embeddings, text_lengths):
        text_embeddings = torch.nn.utils.rnn.pack_padded_sequence(text_embeddings, text_lengths, batch_first=True)
        gru_out, _ = self.gru(text_embeddings, self.hidden)        # [seq_len, batch_size, 2 * hidden_dim]
        gru_out = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=0.0, total_length=None)[0]
        return gru_out


if __name__ == '__main__':

    encoder = GRUEncoder(5, 3)
    text = torch.randn(1, 10, 5)
    length = torch.LongTensor([7])
    print("text")
    print(text)
    out = encoder(text, length)
    print("out")
    print(out)
