import torch
import torch.optim as optim
import numpy as np
import random
import os

from config import cfg
from utils.data_loader import Gen_Data_loader, Dis_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import Rollout

#########################################################################################
# Hyper-parameters (caricati da config.yaml)
#########################################################################################
SEED = cfg['seed']
BATCH_SIZE = cfg['training']['batch_size']
SEQ_LENGTH = cfg['training']['seq_length']
START_TOKEN = cfg['training']['start_token']
VOCAB_SIZE = cfg['training']['vocab_size']

# Generator Params
GEN_EMB_DIM = cfg['generator']['emb_dim']
GEN_HIDDEN_DIM = cfg['generator']['hidden_dim']
PRE_EPOCH_NUM = cfg['generator']['pretrain_epochs']

# Discriminator Params
DIS_EMB_DIM = cfg['discriminator']['emb_dim']
DIS_FILTER_SIZES = cfg['discriminator']['filter_sizes']
DIS_NUM_FILTERS = cfg['discriminator']['num_filters']

# Adversarial Training Params
ADV_TOTAL_BATCH = cfg['adversarial']['total_batches']
ROLLOUT_NUM = cfg['adversarial']['rollout_num']

# Files
POSITIVE_FILE = cfg['paths']['positive_file']
NEGATIVE_FILE = cfg['paths']['negative_file']
EVAL_FILE = cfg['paths']['eval_file']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(model, batch_size, generated_num, output_file):
    """
    Fa generare al modello 'generated_num' payload e li salva su file testuale.
    """
    model.eval() # Modalità inferenza
    generated_samples = []
    
    with torch.no_grad():
        for _ in range(int(generated_num / batch_size)):
            samples, _ = model.sample(batch_size, device)
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
                          sequence_length=SEQ_LENGTH,
                          start_token=START_TOKEN).to(device)
                          
    discriminator = Discriminator(num_classes=2, vocab_size=VOCAB_SIZE, emb_dim=DIS_EMB_DIM, 
                                  filter_sizes=DIS_FILTER_SIZES, num_filters=DIS_NUM_FILTERS,
                                  dropout_prob=cfg['discriminator']['dropout_prob']).to(device)

    # Ottimizzatori PyTorch
    gen_optimizer = optim.Adam(generator.parameters(), lr=cfg['generator']['learning_rate'])
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=cfg['discriminator']['learning_rate'])

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
            x_batch = torch.tensor(gen_data_loader.next_batch(), dtype=torch.long).to(device)

            gen_optimizer.zero_grad()
            
            # Forward pass: prevede i prossimi token
            predictions = generator(x_batch)
            
            # Loss Cross-Entropy
            loss = generator.pretrain_loss(predictions, x_batch)
            loss.backward()
            
            # Gradient clipping (essenziale per LSTM)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg['generator']['gradient_clip'])
            gen_optimizer.step()
            
            epoch_loss += loss.item()

        if epoch % cfg['training']['print_every'] == 0:
            print(f'Epoch Pre-Train Gen {epoch}/{PRE_EPOCH_NUM} - Loss (NLL): {epoch_loss/gen_data_loader.num_batch:.4f}')

    #########################################################################################
    # FASE 2: PRE-TRAINING DEL DISCRIMINATORE
    #########################################################################################
    print('\n[FASE 2] Inizio Pre-Training del Discriminatore...')
    
    for d_step in range(cfg['discriminator']['pretrain_steps']):
        # Genera payload falsi e uniscili a quelli veri
        generate_samples(generator, BATCH_SIZE, cfg['adversarial']['generated_samples'], NEGATIVE_FILE)
        dis_data_loader.load_train_data(POSITIVE_FILE, NEGATIVE_FILE)
        
        for _ in range(cfg['discriminator']['inner_epochs']):
            dis_data_loader.reset_pointer()
            for _ in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                
                # Converti in tensori
                x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
                # y_batch nel dataloader originale restituiva one-hot [0,1], a noi serve l'indice di classe (0 o 1)
                y_tensor = torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.long).to(device)

                dis_optimizer.zero_grad()
                logits = discriminator(x_tensor)
                
                loss = discriminator.compute_loss(logits, y_tensor)
                loss.backward()
                dis_optimizer.step()
                
        if d_step % cfg['training']['print_every'] == 0:
            print(f'Discriminator Pre-train Step {d_step}/{cfg["discriminator"]["pretrain_steps"]} - Loss: {loss.item():.4f}')


    #########################################################################################
    # FASE 3: ADDESTRAMENTO AVVERSARIO (RL POLICY GRADIENT)
    #########################################################################################
    rollout = Rollout(generator, update_rate=cfg['adversarial']['rollout_update_rate'])

    print('\n[FASE 3] Inizio Addestramento Avversario (SeqGAN)...')
    for adv_epoch in range(ADV_TOTAL_BATCH):
        
        # 1. Addestra Generatore per 1 step
        for _ in range(cfg['adversarial']['generator_steps']):
            # Genera sequenze e log_probs
            samples, log_probs = generator.sample(BATCH_SIZE, device)
            
            # Calcola le Reward tramite Rollout MC Search e Discriminatore
            rewards = rollout.get_reward(samples, ROLLOUT_NUM, discriminator)
            
            # Applica Policy Gradient (REINFORCE)
            gen_optimizer.zero_grad()
            g_loss = generator.adversarial_loss(log_probs, rewards)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg['generator']['gradient_clip'])
            gen_optimizer.step()

        # Aggiorna la rete copia del Rollout (Soft Update)
        rollout.update_params(generator)

        # 2. Addestra Discriminatore
        for _ in range(cfg['adversarial']['discriminator_steps']):
            generate_samples(generator, BATCH_SIZE, cfg['adversarial']['generated_samples'], NEGATIVE_FILE)
            dis_data_loader.load_train_data(POSITIVE_FILE, NEGATIVE_FILE)
            
            for _ in range(cfg['discriminator']['inner_epochs']):
                dis_data_loader.reset_pointer()
                for _ in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
                    y_tensor = torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.long).to(device)

                    dis_optimizer.zero_grad()
                    logits = discriminator(x_tensor)
                    d_loss = discriminator.compute_loss(logits, y_tensor)
                    d_loss.backward()
                    dis_optimizer.step()

        if adv_epoch % cfg['training']['print_every'] == 0:
            print(f'Adversarial Epoch {adv_epoch}/{ADV_TOTAL_BATCH} - G_Loss: {g_loss.item():.4f} | D_Loss: {d_loss.item():.4f}')

    print("\n--- ADDESTRAMENTO COMPLETATO! ---")
    
    # Salva i pesi finali del modello
    torch.save(generator.state_dict(), cfg['paths']['model_output'])
    print(f"Modello salvato in '{cfg['paths']['model_output']}'")

if __name__ == '__main__':
    main()