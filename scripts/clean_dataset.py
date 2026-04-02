import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg

def crea_dataset_toy(percorso_attacchi, percorso_kaggle, cartella_output="."):
    """
    Estrae payload da un file di testo e query benigne da un CSV, 
    creando tre CSV di output bilanciati.
    """
    print("Inizio la creazione del dataset...")
    
    # 1. ESTREARRE GLI ATTACCHI (Label = 1)
    attacchi = []
    try:
        with open(percorso_attacchi, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                linea_pulita = line.strip()
                # Evitiamo di inserire righe vuote
                if linea_pulita:
                    attacchi.append(linea_pulita)
    except FileNotFoundError:
        print(f"Errore: File non trovato -> {percorso_attacchi}")
        return

    # Creiamo un DataFrame per gli attacchi
    df_attacchi = pd.DataFrame({"Query": attacchi, "Label": 1})
    num_attacchi = len(df_attacchi)
    print(f"Estratti {num_attacchi} payload malevoli da {percorso_attacchi}")

    # 2. ESTRARRE IL TRAFFICO NORMALE (Label = 0)
    try:
        # Nota: assicurati che i nomi delle colonne del CSV di Kaggle siano corretti.
        # Spesso sono "Query" o "Sentence" e "Label".
        df_kaggle = pd.read_csv(percorso_kaggle)
    except FileNotFoundError:
        print(f"Errore: File non trovato -> {percorso_kaggle}")
        return

    # Filtriamo solo il traffico normale (Label == 0)
    df_normale_totale = df_kaggle[df_kaggle['Label'] == 0]
    
    # Controlliamo se ci sono abbastanza query normali
    num_normali_disponibili = len(df_normale_totale)
    if num_normali_disponibili < num_attacchi:
        print(f"Attenzione: Ci sono solo {num_normali_disponibili} query normali.")
        print(f"Riduco il numero di attacchi a {num_normali_disponibili} per bilanciare.")
        df_attacchi = df_attacchi.sample(n=num_normali_disponibili, random_state=42)
        num_attacchi = num_normali_disponibili
    
    # Estraiamo un numero casuale di query normali ESATTAMENTE uguale al numero di attacchi
    df_normale = df_normale_totale.sample(n=num_attacchi, random_state=42)
    print(f"Estratte {num_attacchi} query benigne da {percorso_kaggle}")

    # 3. CREARE IL DATASET COMBINATO
    # Concateniamo e mescoliamo (shuffle) per evitare che il modello impari un ordine
    df_toy = pd.concat([df_attacchi, df_normale]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. SALVARE I FILE
    percorso_toy = os.path.join(cartella_output, cfg['paths']['generated_dataset'])
    percorso_toy_attack = os.path.join(cartella_output, cfg['paths']['generated_attacks'])
    percorso_toy_normal = os.path.join(cartella_output, cfg['paths']['generated_normal'])

    df_toy.to_csv(percorso_toy, index=False)
    df_attacchi.to_csv(percorso_toy_attack, index=False)
    df_normale.to_csv(percorso_toy_normal, index=False)

    print("\nOperazione completata con successo! File creati:")
    print(f"- {percorso_toy} (Totale: {len(df_toy)} righe)")
    print(f"- {percorso_toy_attack} (Totale: {len(df_attacchi)} righe)")
    print(f"- {percorso_toy_normal} (Totale: {len(df_normale)} righe)")

# Esecuzione dello script
if __name__ == "__main__":
    # Aggiorna questi percorsi se i file si trovano in una directory diversa
    FILE_ATTACCHI = cfg['paths']['raw_attacks']
    FILE_KAGGLE = cfg['paths']['raw_kaggle']
    
    crea_dataset_toy(FILE_ATTACCHI, FILE_KAGGLE)