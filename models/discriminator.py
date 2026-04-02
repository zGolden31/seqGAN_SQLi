import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Livello Highway (http://arxiv.org/abs/1505.00387).
    Permette alle reti profonde di far fluire l'informazione non modificata,
    mitigando il problema della scomparsa del gradiente.
    """
    def __init__(self, size, num_layers=1):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        
        # Le porte di trasformazione (gate) e non-lineari
        self.transform_gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            # t = sigmoid(Wy + b)
            t = torch.sigmoid(self.transform_gate[i](x))
            # g = ReLU(Wy + b)
            g = F.relu(self.nonlinear[i](x))
            
            # z = t * g + (1 - t) * x
            x = t * g + (1. - t) * x
        return x

class Discriminator(nn.Module):
    """
    Una CNN per la classificazione del testo (ispirata alla Kim CNN),
    utilizzata come Discriminatore in una SeqGAN classica.
    """
    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters,
                 dropout_prob=0.7):
        """
        num_classes: 2 (Reale vs Falso)
        vocab_size: Dimensione del vocabolario (max_vocab_size)
        emb_dim: Dimensione del vettore di embedding
        filter_sizes: Lista di dimensioni dei kernel (es. [2, 3, 4, 5] per n-grams)
        num_filters: Lista col numero di filtri per ogni kernel (es. [100, 100, 100, 100])
        """
        super(Discriminator, self).__init__()
        
        # Embedding Layer per i token
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)

        # Creazione dei livelli Convoluzionali 1D
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim,
                      out_channels=nf,
                      kernel_size=fs)
            for fs, nf in zip(filter_sizes, num_filters)
        ])

        # Calcoliamo il totale dei filtri estratti per dimensionare i livelli successivi
        num_filters_total = sum(num_filters)

        # Livello Highway, Dropout e Output Lineare
        self.highway = Highway(num_filters_total, num_layers=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(num_filters_total, num_classes)

    def forward(self, x):
        """
        x: tensore di interi di shape [batch_size, sequence_length]
        """
        
        # 1. Embedding [batch_size, sequence_length, emb_dim]
        word_emb = self.word_embedding(x)
        
        # In PyTorch, nn.Conv1d si aspetta i dati nel formato: [batch_size, channels, sequence_length]
        # Quindi scambiamo la dimensione dei canali (gli embedding) con quella della sequenza
        emb = word_emb.permute(0, 2, 1) # [batch_size, emb_dim, seq_len]

        # 4. Convoluzione e Max Pooling
        pooled_outputs = []
        for conv in self.convs:
            # Convoluzione seguita da ReLU
            c = F.relu(conv(emb)) # Shape: [batch, num_filters, seq_len - filter_size + 1]
            
            # Max Pooling sull'intera sequenza temporale per estrarre la feature più rilevante
            p = F.max_pool1d(c, kernel_size=c.size(2)) # Shape: [batch, num_filters, 1]
            p = p.squeeze(2) # Rimuoviamo l'ultima dimensione: [batch, num_filters]
            
            pooled_outputs.append(p)

        # 5. Combinazione di tutte le feature estratte dai vari n-grams
        h_pool_flat = torch.cat(pooled_outputs, dim=1) # [batch_size, num_filters_total]

        # 6. Highway e Dropout
        h_highway = self.highway(h_pool_flat)
        h_drop = self.dropout(h_highway)

        # 7. Classificazione Finale
        logits = self.fc(h_drop) # [batch_size, num_classes]
        
        return logits

    def compute_loss(self, logits, labels):
        """
        Metodo di utilità per calcolare la loss.
        Equivalente a tf.nn.softmax_cross_entropy_with_logits.
        """
        # labels qui si aspetta gli indici di classe (0 per falso, 1 per vero),
        # non la codifica one-hot come in TF. Se usi one-hot [0,1], usa torch.argmax(labels, dim=1)
        return F.cross_entropy(logits, labels)