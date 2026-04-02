import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Generator(nn.Module):
    def __init__(self, num_emb, emb_dim, hidden_dim, sequence_length, num_conditions, start_token):
        """
        num_emb: Dimensione del vocabolario (max_vocab_size)
        emb_dim: Dimensione del vettore di embedding
        hidden_dim: Dimensione della memoria interna della LSTM
        sequence_length: Lunghezza fissa (es. 50)
        num_conditions: Numero di DBMS diversi (es. 4 per MySQL, Oracle, ecc.)
        start_token: L'ID del token <SOS> (es. 2)
        """
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token

        # Equivalente a self.g_embeddings
        self.word_embedding = nn.Embedding(num_emb, emb_dim)
        
        # NUOVO PER CGAN: Embedding per la condizione (es. tipo di DBMS)
        self.cond_embedding = nn.Embedding(num_conditions, emb_dim)

        # Equivalente a create_recurrent_unit()
        # L'input è il doppio perché concateniamo il token e la condizione
        self.lstm_cell = nn.LSTMCell(emb_dim * 2, hidden_dim)

        # Equivalente a create_output_unit()
        self.linear = nn.Linear(hidden_dim, num_emb)

    def forward(self, x, condition):
        """
        FASE DI PRE-TRAINING (Supervised Learning - MLE)
        Equivalente a _pretrain_recurrence e self.g_predictions.
        Usa il "Teacher Forcing": diamo in pasto la vera sequenza per prevedere il token successivo.
        """
        batch_size = x.size(0)
        device = x.device

        # Stati iniziali h0 e c0 (memoria della LSTM)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        # Costruiamo l'input: [START_TOKEN, Token_1, Token_2, ..., Token_N-1]
        start_tokens = torch.full((batch_size, 1), self.start_token, dtype=torch.long).to(device)
        gen_input = torch.cat([start_tokens, x[:, :-1]], dim=1)

        # Calcoliamo gli embedding
        word_emb = self.word_embedding(gen_input) # [batch, seq_len, emb_dim]
        cond_emb = self.cond_embedding(condition).unsqueeze(1).expand(-1, self.sequence_length, -1)

        # Concateniamo per formare l'input della rete
        lstm_input = torch.cat([word_emb, cond_emb], dim=-1)

        logits_list = []

        # Il while_loop di TF diventa un semplice for in PyTorch
        for i in range(self.sequence_length):
            h_t, c_t = self.lstm_cell(lstm_input[:, i, :], (h_t, c_t))
            logits = self.linear(h_t)
            logits_list.append(logits)

        # Impiliamo le previsioni: [batch_size, seq_length, vocab_size]
        return torch.stack(logits_list, dim=1)

    def sample(self, batch_size, condition):
        """
        FASE ADVERSARIAL (Unsupervised Training / Generazione)
        Equivalente a _g_recurrence e self.gen_x.
        La rete genera un token alla volta, e il token generato diventa l'input per lo step successivo.
        """
        device = condition.device
        
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        # Si parte sempre dal token <SOS>
        x_t = torch.full((batch_size,), self.start_token, dtype=torch.long).to(device)
        cond_emb = self.cond_embedding(condition) # La condizione non cambia durante la generazione

        samples = []
        log_probs = []

        for i in range(self.sequence_length):
            word_emb = self.word_embedding(x_t)
            lstm_input = torch.cat([word_emb, cond_emb], dim=-1)

            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))
            logits = self.linear(h_t)

            # Equivalente a tf.multinomial(log_prob, 1)
            # Trasformiamo i logits in probabilità e campioniamo il token successivo
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            x_t = dist.sample() # Prende una decisione (un ID del token)

            samples.append(x_t)
            log_probs.append(dist.log_prob(x_t)) # Salviamo il log_prob per calcolare la Reward dopo

        # Restituiamo i payload generati e le loro probabilità logaritmiche
        return torch.stack(samples, dim=1), torch.stack(log_probs, dim=1)

    # --- FUNZIONI DI LOSS ---

    def pretrain_loss(self, predictions, targets):
        """
        Equivalente a self.pretrain_loss
        Cross Entropy classica per imparare la sintassi di base.
        """
        # Trasformiamo per adattarli alla funzione di loss di PyTorch
        predictions = predictions.view(-1, self.num_emb)
        targets = targets.contiguous().view(-1)
        return F.cross_entropy(predictions, targets)

    def adversarial_loss(self, log_probs, rewards):
        """
        Equivalente a self.g_loss
        Algoritmo REINFORCE (Policy Gradient). Moltiplichiamo la certezza della rete (log_probs)
        per il voto del discriminatore (rewards).
        """
        # Minimizziamo il negativo del valore atteso (quindi massimizziamo la reward)
        loss = -torch.sum(log_probs * rewards)
        return loss