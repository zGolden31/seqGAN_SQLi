import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

class Rollout:
    def __init__(self, generator, update_rate):
        """
        generator: L'istanza della classe Generator che stiamo addestrando.
        update_rate: Il tasso di aggiornamento per l'Exponential Moving Average (EMA).
        """
        # Creiamo una copia esatta e indipendente del Generatore
        self.rollout_gen = copy.deepcopy(generator)
        self.rollout_gen.eval() # Il Rollout non apprende, simula e basta
        self.update_rate = update_rate

    def update_params(self, generator):
        """
        Equivalente a update_recurrent_unit e update_output_unit del codice TF.
        Aggiorna i pesi del Rollout facendoli convergere lentamente verso quelli
        del Generatore principale (Soft Update).
        """
        rollout_dict = self.rollout_gen.state_dict()
        gen_dict = generator.state_dict()
        
        for key in rollout_dict:
            # W_rollout = update_rate * W_rollout + (1 - update_rate) * W_gen
            rollout_dict[key] = self.update_rate * rollout_dict[key] + (1 - self.update_rate) * gen_dict[key]
            
        self.rollout_gen.load_state_dict(rollout_dict)

    def sample_from_prefix(self, prefix, total_length, condition):
        """
        Equivalente ai cicli _g_recurrence_1 (che legge il prefisso) e 
        _g_recurrence_2 (che genera il resto) del codice TF.
        """
        batch_size = prefix.size(0)
        device = prefix.device
        t_prefix = prefix.size(1)

        # Inizializza memoria della LSTM
        h_t = torch.zeros(batch_size, self.rollout_gen.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.rollout_gen.hidden_dim).to(device)
        cond_emb = self.rollout_gen.cond_embedding(condition)

        # 1. Carica la memoria della rete leggendo il prefisso noto
        start_tokens = torch.full((batch_size, 1), self.rollout_gen.start_token, dtype=torch.long).to(device)
        gen_input = torch.cat([start_tokens, prefix], dim=1) # [<SOS>, token_1, token_2...]

        # Passiamo tutto il prefisso nella LSTM senza generare nulla
        for i in range(t_prefix):
            word_emb = self.rollout_gen.word_embedding(gen_input[:, i])
            lstm_input = torch.cat([word_emb, cond_emb], dim=-1)
            h_t, c_t = self.rollout_gen.lstm_cell(lstm_input, (h_t, c_t))

        # L'ultimo token elaborato è il punto di partenza per il rollout
        x_t = gen_input[:, -1]
        generated_tokens = []

        # 2. Genera (Rollout) la parte mancante della frase
        for i in range(t_prefix, total_length):
            word_emb = self.rollout_gen.word_embedding(x_t)
            lstm_input = torch.cat([word_emb, cond_emb], dim=-1)

            h_t, c_t = self.rollout_gen.lstm_cell(lstm_input, (h_t, c_t))
            logits = self.rollout_gen.linear(h_t)

            # Campiona il token successivo
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            x_t = dist.sample()

            generated_tokens.append(x_t)

        # Unisce il prefisso alla parte inventata dal Rollout
        if generated_tokens:
            suffix = torch.stack(generated_tokens, dim=1)
            complete_seq = torch.cat([prefix, suffix], dim=1)
        else:
            complete_seq = prefix

        return complete_seq

    def get_reward(self, x, rollout_num, discriminator, condition):
        """
        Equivalente a get_reward. 
        Calcola il voto del discriminatore per ogni singolo step della generazione.
        x: I payload generati [batch_size, seq_len]
        condition: Il DBMS bersaglio [batch_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device

        rewards = torch.zeros(batch_size, seq_len).to(device)

        # Disabilitiamo i gradienti: il Rollout è solo un calcolo di supporto
        with torch.no_grad():
            for i in range(rollout_num):
                # Cicliamo su ogni token della frase (es. da 1 a 49)
                for t in range(1, seq_len):
                    # Prendiamo la frase fino al punto 't'
                    prefix = x[:, :t]

                    # Chiediamo al Rollout di simulare la fine della frase
                    complete_seq = self.sample_from_prefix(prefix, seq_len, condition)

                    # Chiediamo al Discriminatore se la frase inventata ha senso
                    logits = discriminator(complete_seq, condition)
                    
                    # Estraiamo la probabilità che il payload sia REALE (classe 1)
                    prob_real = F.softmax(logits, dim=1)[:, 1]
                    
                    rewards[:, t-1] += prob_real

                # Per l'ultimo token (t = seq_len) non c'è nulla da simulare
                logits = discriminator(x, condition)
                prob_real = F.softmax(logits, dim=1)[:, 1]
                rewards[:, seq_len-1] += prob_real

        # Facciamo la media dei voti ottenuti nei vari Rollout
        rewards = rewards / (1.0 * rollout_num)
        
        return rewards