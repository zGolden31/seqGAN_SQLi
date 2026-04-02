import os
import pandas as pd

# Esempio: se la classe è in utils/tokenizer.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tokenizer import SQLiTokenizer
from config import cfg

def preprocess_pipeline():
    # 1. SETUP DEI PERCORSI
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Crea la cartella processed se non esiste
    os.makedirs(processed_dir, exist_ok=True)

    file_attacchi_csv = os.path.join(base_dir, cfg['paths']['generated_dataset'])
    file_normali_csv = os.path.join(base_dir, cfg['paths']['generated_normal'])
    
    out_attacchi_txt = os.path.join(base_dir, cfg['paths']['processed_attacks'])
    out_normali_txt = os.path.join(base_dir, cfg['paths']['processed_normal'])
    out_vocab = os.path.join(base_dir, cfg['paths']['vocab_file'])

    print("--- INIZIO PRE-PROCESSING ---")

    # 2. CARICAMENTO DATI
    try:
        df_attacchi = pd.read_csv(file_attacchi_csv)
        df_normali = pd.read_csv(file_normali_csv)
    except FileNotFoundError as e:
        print(f"Errore: File non trovato. Assicurati di aver generato i dataset toy. Dettagli: {e}")
        return

    # Pulisci eventuali righe vuote (NaN)
    attacchi = df_attacchi['Query'].dropna().astype(str).tolist()
    normali = df_normali['Query'].dropna().astype(str).tolist()

    print(f"Caricati {len(attacchi)} attacchi e {len(normali)} query benigne.")

    # 3. INIZIALIZZAZIONE E ADDESTRAMENTO TOKENIZER
    tokenizer = SQLiTokenizer(max_seq_length=cfg['training']['seq_length'], max_vocab_size=cfg['training']['vocab_size'])
    
    # Uniamo i testi per creare un vocabolario globale condiviso
    tutti_i_testi = attacchi + normali
    print("Costruzione del vocabolario in corso (potrebbe richiedere qualche secondo)...")
    tokenizer.fit(tutti_i_testi)

    # 4. SALVATAGGIO DEL VOCABOLARIO
    tokenizer.save(out_vocab)

    # 5. CONVERSIONE E SALVATAGGIO: ATTACCHI
    print(f"Codifica e salvataggio attacchi in {out_attacchi_txt}...")
    with open(out_attacchi_txt, 'w') as f_out:
        for payload in attacchi:
            encoded_ids = tokenizer.encode(payload)
            # Converte la lista di interi in una stringa separata da spazi
            linea_testo = " ".join(map(str, encoded_ids))
            f_out.write(linea_testo + "\n")

    # 6. CONVERSIONE E SALVATAGGIO: NORMALI
    print(f"Codifica e salvataggio traffico normale in {out_normali_txt}...")
    with open(out_normali_txt, 'w') as f_out:
        for payload in normali:
            encoded_ids = tokenizer.encode(payload)
            linea_testo = " ".join(map(str, encoded_ids))
            f_out.write(linea_testo + "\n")

    print("--- PRE-PROCESSING COMPLETATO CON SUCCESSO ---")
    print(f"I dati sono pronti per essere caricati da GenDataLoader e DisDataLoader!")

if __name__ == "__main__":
    preprocess_pipeline()