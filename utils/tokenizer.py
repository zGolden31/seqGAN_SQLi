import sqlparse
from sqlparse.sql import TokenList
import json

class SQLiTokenizer:
    def __init__(self, max_seq_length=50):
        self.max_seq_length = max_seq_length
        # Token speciali
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def _flatten_tokens(self, tokens):
        """Estrae ricorsivamente i token da strutture annidate."""
        flat_list = []
        for token in tokens:
            if isinstance(token, TokenList):
                flat_list.extend(self._flatten_tokens(token.tokens))
            else:
                # Escludiamo gli spazi vuoti superflui per pulire il payload
                if not token.is_whitespace:
                    flat_list.append(token.value.upper()) # Normalizziamo in maiuscolo
        return flat_list

    def fit(self, payloads):
        """Costruisce il vocabolario basandosi su una lista di payload."""
        for p in payloads:
            parsed = sqlparse.parse(p)
            tokens = self._flatten_tokens(parsed[0].tokens)
            for t in tokens:
                if t not in self.vocab:
                    new_id = len(self.vocab)
                    self.vocab[t] = new_id
                    self.id_to_token[new_id] = t
        print(f"Vocabolario creato: {len(self.vocab)} token unici.")

    def encode(self, payload):
        """Converte una stringa in una lista di numeri (IDs)."""
        parsed = sqlparse.parse(payload)
        tokens = self._flatten_tokens(parsed[0].tokens)
        
        # Aggiungiamo Start e End of Sentence
        encoded = [self.vocab["<SOS>"]]
        for t in tokens:
            encoded.append(self.vocab.get(t, self.vocab["<UNK>"]))
        encoded.append(self.vocab["<EOS>"])
        
        # Padding o Troncamento
        if len(encoded) < self.max_seq_length:
            encoded += [self.vocab["<PAD>"]] * (self.max_seq_length - len(encoded))
        else:
            encoded = encoded[:self.max_seq_length]
            
        return encoded

    def decode(self, ids):
        """Converte una lista di ID in testo."""
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids if i not in [0, 2, 3]])
    
    def save(self, path):
        """Salva vocab su disco — necessario per riusarlo in train.py e generate.py."""
        with open(path, "w") as f:
            json.dump({
                "max_seq_length": self.max_seq_length,
                "max_vocab_size": self.max_vocab_size,
                "vocab": self.vocab
            }, f, indent=2)
        print(f"Tokenizer salvato in {path}")    

    @classmethod
    def load(cls, path):
        """Carica il tokenizer da un file JSON salvato."""
        import json # Assicurati che sia importato in cima al file
        with open(path, "r") as f:
            data = json.load(f)
            
        # Crea una nuova istanza usando i parametri salvati
        # Uso .get() per max_vocab_size nel caso in cui non fosse definito nei vecchi salvataggi
        tokenizer = cls(max_seq_length=data["max_seq_length"])
        if "max_vocab_size" in data:
            tokenizer.max_vocab_size = data["max_vocab_size"]
            
        # Ripristina i dizionari
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        print(f"Tokenizer caricato da {path} (Vocab size: {len(tokenizer.vocab)})")
        return tokenizer

