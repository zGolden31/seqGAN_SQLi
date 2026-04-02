import numpy as np


class GenDataLoader:
    """
    Data loader per il Generatore (pre-training e campionamento).
    Legge file di testo dove ogni riga è una sequenza di token ID separati da spazio.
    """

    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.token_stream = []

    def create_batches(self, data_file):
        """Carica le sequenze dal file e le suddivide in batch."""
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parse_line = [int(x) for x in line.split()]
                if len(parse_line) == self.seq_length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        """Restituisce il prossimo batch e avanza il puntatore ciclicamente."""
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataLoader:
    """
    Data loader per il Discriminatore.
    Carica esempi positivi (payload SQLi reali) e negativi (sequenze generate dal generatore),
    assegna le label e prepara i batch mescolati.
    """

    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        """
        Carica i dati di training per il discriminatore.

        Args:
            positive_file: file con sequenze reali (SQLi tokenizzati) — label [0, 1]
            negative_file: file con sequenze generate dal generatore  — label [1, 0]
        """
        positive_examples = []
        negative_examples = []

        with open(positive_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parse_line = [int(x) for x in line.split()]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)

        with open(negative_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parse_line = [int(x) for x in line.split()]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)

        # Label one-hot: [0, 1] = reale (positivo), [1, 0] = generato (negativo)
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], axis=0)

        # Shuffle
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split in batch
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        """Restituisce il prossimo batch (sequenze, label) e avanza il puntatore."""
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0