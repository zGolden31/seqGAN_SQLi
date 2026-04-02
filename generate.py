import torch
import os
import sys

# Assicuriamoci che Python trovi i moduli
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import cfg
from utils.tokenizer import SQLiTokenizer
from models.generator import Generator

# --- CONFIGURAZIONE (caricata da config.yaml) ---
SEQ_LENGTH = cfg['training']['seq_length']
START_TOKEN = cfg['training']['start_token']
GEN_EMB_DIM = cfg['generator']['emb_dim']
GEN_HIDDEN_DIM = cfg['generator']['hidden_dim']

def load_generator(vocab_size, model_path, device):
    """Inizializza il modello e carica i pesi salvati."""
    generator = Generator(
        num_emb=vocab_size, 
        emb_dim=GEN_EMB_DIM, 
        hidden_dim=GEN_HIDDEN_DIM, 
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN
    ).to(device)
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval() # Imposta il modello in modalità inferenza
        print(f"Pesi del modello caricati con successo da {model_path}")
    else:
        raise FileNotFoundError(f"Impossibile trovare il modello in {model_path}. Hai avviato train.py?")
        
    return generator

def generate_payloads(num_payloads, generator, tokenizer, device):
    """Genera e decodifica i payload sintetici."""
    print(f"\nGenerazione di {num_payloads} payload sintetici...")
    
    # Disabilitiamo i gradienti per la generazione
    with torch.no_grad():
        # Generiamo le sequenze di ID
        samples, _ = generator.sample(num_payloads, device)
        
    # Convertiamo i tensori in liste di Python
    samples_list = samples.cpu().numpy().tolist()
    
    payloads_testuali = []
    for seq in samples_list:
        # Il nostro metodo decode ignora in automatico gli ID di <PAD>, <SOS> e <EOS>
        testo = tokenizer.decode(seq)
        payloads_testuali.append(testo)
        
    return payloads_testuali

def main():
    # Setup del device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo Device: {device}")
    
    # Percorsi dei file
    vocab_path = cfg['paths']['vocab_file']
    model_path = cfg['paths']['model_output']
    output_path = cfg['paths']['synthetic_output']
    
    # 1. Caricamento del Tokenizer
    if not os.path.exists(vocab_path):
        print(f"Errore: Vocabolario {vocab_path} non trovato. Esegui preprocess_data.py prima.")
        return
        
    tokenizer = SQLiTokenizer.load(vocab_path)
    
    # 2. Caricamento del Generatore
    generator = load_generator(len(tokenizer.vocab), model_path, device)

    
    # 3. Parametri di generazione
    NUM_PAYLOADS = cfg['generation']['num_payloads']
    
    # 4. Generazione
    payloads = generate_payloads(NUM_PAYLOADS, generator, tokenizer, device)
    
    # 5. Stampa e Salvataggio
    print("\n--- RISULTATI GENERATI ---")
    for i, p in enumerate(payloads):
        # Ripuliamo gli spazi extra prima della stampa
        clean_p = " ".join(p.split())
        print(f"{i+1:02d}: {clean_p}")
        
    # Salva su file
    with open(output_path, "w") as f:
        for p in payloads:
            f.write(" ".join(p.split()) + "\n")
            
    print(f"\nI payload sono stati salvati in {output_path}")

if __name__ == "__main__":
    main()