import torch
import torch.optim as optim
import numpy as np
import random
import os

from utils.data_loader import Gen_Data_loader, Dis_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import Rollout

#########################################################################################
# Hyper-parameters
#########################################################################################
SEED = 42
BATCH_SIZE = 64
SEQ_LENGTH = 50
START_TOKEN = 2  # <SOS> in base al tuo SQLiTokenizer
VOCAB_SIZE = 5000 # Deve combaciare col tokenizer
NUM_CONDITIONS = 4 # Esempio: 0=MySQL, 1=PostgreSQL, 2=MSSQL, 3=Oracle

# Generator Params
GEN_EMB_DIM = 32
GEN_HIDDEN_DIM = 32
PRE_EPOCH_NUM = 120 # Quante epoche per il Pre-training MLE

# Discriminator Params
DIS_EMB_DIM = 64
DIS_FILTER_SIZES = [2, 3, 4, 5, 6, 7] # N-grams (bi-grams, tri-grams, ecc.)
DIS_NUM_FILTERS = [100, 100, 100, 100, 100, 100]

# Adversarial Training Params
ADV_TOTAL_BATCH = 200
ROLLOUT_NUM = 16

# Files
POSITIVE_FILE = 'data/processed/real_attack_data.txt'
NEGATIVE_FILE = 'data/processed/generator_sample.txt'
EVAL_FILE = 'data/processed/eval_file.txt'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(model, batch_size, generated_num, output_file, num_conditions):
    """
    Fa generare al modello 'generated_num' payload e li salva su file testuale.
    """
    model.eval() # Modalità inferenza
    generated_samples = []
    
    with torch.no_grad():
        for _ in range(int(generated_num / batch_size)):
            # Per generare, scegliamo una condizione casuale o fissa.
            # Qui simuliamo la generazione distribuita tra i vari DBMS
            cond = torch.randint(0, num_conditions, (batch_size,)).to(device)
            samples, _ = model.sample(batch_size, cond)
            generated_samples.extend(samples.cpu().numpy())

    with open(output_file, 'w') as fout:
        for payload in generated_samples:
            buffer = ' '.join([str(x) for x in payload]) + '\n'
            fout.write(buffer)
    model.train() # Torniamo in training mode

def main():
    # 1. Setup e Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"Utilizzo Device: {device}")

    # 2. Inizializzazione Data Loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, seq_length=SEQ_LENGTH)
    gen_data_loader.create_batches(POSITIVE_FILE) # Dati reali per il pre-train del Gen
    
    dis_data_loader = Dis_dataloader(BATCH_SIZE, seq_length=SEQ_LENGTH)

    # 3. Inizializzazione Modelli
    generator = Generator(num_emb=VOCAB_SIZE, emb_dim=GEN_EMB_DIM, hidden_dim=GEN_HIDDEN_DIM, 
                          sequence_length=SEQ_LENGTH, num_conditions=NUM_CONDITIONS, 
                          start_token=START_TOKEN).to(device)
                          
    discriminator = Discriminator(num_classes=2, vocab_size=VOCAB_SIZE, emb_dim=DIS_EMB_DIM, 
                                  filter_sizes=DIS_FILTER_SIZES, num_filters=DIS_NUM_FILTERS, 
                                  num_conditions=NUM_CONDITIONS).to(device)

    # Ottimizzatori PyTorch
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.01)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Assicuriamoci che la cartella esista
    os.makedirs('data/processed', exist_ok=True)

    #########################################################################################
    # FASE 1: PRE-TRAINING DEL GENERATORE (MLE)
    #########################################################################################
    print('\n[FASE 1] Inizio Pre-Training MLE del Generatore...')
    for epoch in range(PRE_EPOCH_NUM):
        gen_data_loader.reset_pointer()
        epoch_loss = 0
        
        for _ in range(gen_data_loader.num_batch):
            # Ottieni batch dal loader e simula condizioni causali (poiché non le abbiamo salvate nel txt)
            # In un progetto avanzato, salveresti l'ID del DBMS nel txt e lo caricheresti qui
            x_batch = torch.tensor(gen_data_loader.next_batch(), dtype=torch.long).to(device)
            cond_batch = torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,)).to(device)

            gen_optimizer.zero_grad()
            
            # Forward pass: prevede i prossimi token
            predictions = generator(x_batch, cond_batch)
            
            # Loss Cross-Entropy
            loss = generator.pretrain_loss(predictions, x_batch)
            loss.backward()
            
            # Gradient clipping (essenziale per LSTM)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            gen_optimizer.step()
            
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch Pre-Train Gen {epoch}/{PRE_EPOCH_NUM} - Loss (NLL): {epoch_loss/gen_data_loader.num_batch:.4f}')

    #########################################################################################
    # FASE 2: PRE-TRAINING DEL DISCRIMINATORE
    #########################################################################################
    print('\n[FASE 2] Inizio Pre-Training del Discriminatore...')
    
    for d_step in range(50): # 50 cicli per indurire il discriminatore
        # Genera payload falsi e uniscili a quelli veri
        generate_samples(generator, BATCH_SIZE, 10000, NEGATIVE_FILE, NUM_CONDITIONS)
        dis_data_loader.load_train_data(POSITIVE_FILE, NEGATIVE_FILE)
        
        for _ in range(3): # 3 epoche per ogni ciclo
            dis_data_loader.reset_pointer()
            for _ in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                
                # Converti in tensori
                x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
                # y_batch nel dataloader originale restituiva one-hot [0,1], a noi serve l'indice di classe (0 o 1)
                y_tensor = torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.long).to(device)
                cond_batch = torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,)).to(device)

                dis_optimizer.zero_grad()
                logits = discriminator(x_tensor, cond_batch)
                
                loss = discriminator.compute_loss(logits, y_tensor)
                loss.backward()
                dis_optimizer.step()
                
        if d_step % 10 == 0:
            print(f'Discriminator Pre-train Step {d_step}/50 - Loss: {loss.item():.4f}')


    #########################################################################################
    # FASE 3: ADDESTRAMENTO AVVERSARIO (RL POLICY GRADIENT)
    #########################################################################################
    rollout = Rollout(generator, update_rate=0.8)

    print('\n[FASE 3] Inizio Addestramento Avversario (SeqGAN)...')
    for adv_epoch in range(ADV_TOTAL_BATCH):
        
        # 1. Addestra Generatore per 1 step
        for _ in range(1):
            cond_batch = torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,)).to(device)
            
            # Genera sequenze e log_probs
            samples, log_probs = generator.sample(BATCH_SIZE, cond_batch)
            
            # Calcola le Reward tramite Rollout MC Search e Discriminatore
            rewards = rollout.get_reward(samples, ROLLOUT_NUM, discriminator, cond_batch)
            
            # Applica Policy Gradient (REINFORCE)
            gen_optimizer.zero_grad()
            g_loss = generator.adversarial_loss(log_probs, rewards)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            gen_optimizer.step()

        # Aggiorna la rete copia del Rollout (Soft Update)
        rollout.update_params(generator)

        # 2. Addestra Discriminatore
        for _ in range(5):
            generate_samples(generator, BATCH_SIZE, 10000, NEGATIVE_FILE, NUM_CONDITIONS)
            dis_data_loader.load_train_data(POSITIVE_FILE, NEGATIVE_FILE)
            
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for _ in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
                    y_tensor = torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.long).to(device)
                    cond_batch = torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,)).to(device)

                    dis_optimizer.zero_grad()
                    logits = discriminator(x_tensor, cond_batch)
                    d_loss = discriminator.compute_loss(logits, y_tensor)
                    d_loss.backward()
                    dis_optimizer.step()

        if adv_epoch % 10 == 0:
            print(f'Adversarial Epoch {adv_epoch}/{ADV_TOTAL_BATCH} - G_Loss: {g_loss.item():.4f} | D_Loss: {d_loss.item():.4f}')

    print("\n--- ADDESTRAMENTO COMPLETATO! ---")
    
    # Salva i pesi finali del modello
    torch.save(generator.state_dict(), 'models/generator_final.pth')
    print("Modello salvato in 'models/generator_final.pth'")

if __name__ == '__main__':
    main()