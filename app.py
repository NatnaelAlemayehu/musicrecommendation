from flask import Flask,render_template,Response, redirect, request, url_for
# EDA
import pandas as pd
import numpy as np
np.random.seed(123)
import os

# data visualization
# import matplotlib.pyplot as plt
# import seaborn as sns

# training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm.notebook import tqdm
import pickle


class MusicTrainDataset(Dataset):    

    def __init__(self, ratings, all_SongIDs):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_SongIDs)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_SongIDs):
        # Placeholders to hold the training data
        users, items, labels = [], [], []

        # Set of songs that each user interacts with
        user_item_set = set(zip(ratings['User ID'], ratings['Song ID']))

        # Ratio of negative to positive samples
        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)

        for _ in range(num_negatives):
            negative_item = np.random.choice(all_SongIDs)

            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(all_SongIDs)

            users.append(u)
            items.append(negative_item)
            labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)





class NCF(pl.LightningModule):


    def __init__(self, num_users, num_songs, ratings, all_SongIDs):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_songs, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_SongIDs = all_SongIDs
        
    def forward(self, user_input, song_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        song_embedded = self.item_embedding(song_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, song_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, song_input, labels = batch
        predicted_labels = self(user_input, song_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MusicTrainDataset(self.ratings, self.all_SongIDs), batch_size=512, num_workers=4)




app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():    
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():  

    ratingsfile = os.path.join(app.static_folder, 'test_ratings.csv')
     
    ratings = pd.read_csv(ratingsfile)

    userid = int(request.form.get("userid"))

    modelfile = os.path.join(app.static_folder, 'songrecommender')
   
    loaded_model = pickle.load(open(modelfile, 'rb'))
   
    # Dict of all songs that are interacted with by each user
    user_interacted_songs = ratings.groupby('User ID')['Song ID'].apply(list).to_dict()
    songid = user_interacted_songs[userid][0]

    # Get a list of all songs IDs
    all_SongIDs = ratings['Song ID'].unique()

    top10_songs = []
   
    for (u,i) in tqdm({(94842, 131220)}):  

        interacted_songs = user_interacted_songs[u]      
        not_interacted_songs = set(all_SongIDs) - set(interacted_songs)       
        selected_not_interacted = list(np.random.choice(list(not_interacted_songs), 99))        
        test_songs = selected_not_interacted + [i]        
        predicted_labels = np.squeeze(loaded_model(torch.tensor([u]*100), 
                                            torch.tensor(test_songs)).detach().numpy())          
        top10_songs = [test_songs[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()] 

    return render_template('recommend.html', recommendations= [top10_songs])

if __name__=="__main__":
    app.run(debug=True)



