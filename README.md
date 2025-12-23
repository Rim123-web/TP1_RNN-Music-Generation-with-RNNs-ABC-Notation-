# 🎵 RNN Music Generation (ABC Format)

## Description
Ce projet illustre l’application des **Réseaux de Neurones Récurrents (RNN)** pour générer des séquences musicales au format ABC.  
Le pipeline complet inclut :  
- Prétraitement des partitions musicales  
- Création d’un **dataset PyTorch** adapté aux séquences  
- Conception et entraînement d’un **modèle LSTM**  
- Génération de nouvelles chansons à partir d’une séquence de départ  

Ce projet a un objectif pédagogique pour pratiquer les concepts théoriques des RNN sur un problème concret.

---

## Dataset
Le dataset utilisé provient de [HF Dataset: sander-wood/irishman](https://huggingface.co/datasets/sander-wood/irishman).  
- **Train** : partitions pour l’apprentissage  
- **Validation** : partitions pour l’évaluation  

Chaque partition est un texte ABC contenant notes, rythme, et métadonnées (tonalité, mesure, etc.).

---

##  Installation
Clonez le dépôt et installez les dépendances :


git clone https://github.com/Rim123-web/TP1_RNN-Music-Generation-with-RNNs-ABC-Notation-.git


##  Dépendances principales
<pre>
Python 3.9+
PyTorch
Pandas
Numpy
TensorBoard
tqdm
</pre>

##  Utilisation

### 1️ Prétraitement et création du dataset
<pre>
from preprocessing import vectorize_data, MusicDataset

train_dataset, val_dataset, char2idx, idx2char = vectorize_data(
    'train.json', 
    'validation.json'
)
</pre>

### 2️ Entraînement du modèle
<pre>
from model import MusicRNN, train_model

model = MusicRNN(
    vocab_size=len(char2idx), 
    embedding_dim=256, 
    hidden_size=1024
)

train_losses, val_losses = train_model(
    model, 
    train_loader, 
    val_loader, 
    num_iterations=25, 
    learning_rate=0.005
)
</pre>

### 3️ Génération de musique
<pre>
from generate import generate_music

start_sequence = "X:1\nT:MySong\nM:4/4\nK:C\n"

# Greedy
song_greedy = generate_music(model, start_sequence, char2idx, idx2char, length=200, sample=False)
print("🎵 Generated Song (Greedy):\n", song_greedy)

# Sampling
song_sampled = generate_music(model, start_sequence, char2idx, idx2char, length=200, sample=True, temperature=1.2)
print("🎵 Generated Song (Sampled):\n", song_sampled)
</pre>

##  Résultats
<pre>
Le modèle génère de nouvelles partitions cohérentes avec le style du dataset.

Deux approches possibles :
- Greedy : prend le caractère le plus probable à chaque étape
- Sampling : échantillonne selon les probabilités et un paramètre temperature pour plus de diversité
</pre>
### 📝 PS
- `TP1_RNN.docx` : le TP complet  
- `TP1_RNN.ipynb` : le notebook avec le code et les expérimentations  
- `Rapport_TP1_RNN.pdf` : le rapport détaillé
