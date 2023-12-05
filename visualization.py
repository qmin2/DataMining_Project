import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import argparse
import datetime
import copy
import pandas as pd

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

import modeling
import utils

def main():
    ####### argument 수정하기
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--pooling', default='first', type=str)

    # Hyper parameter
    parser.add_argument('--temperature', default=0.05, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int) # To be checked
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float) 
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--patience', default = 10, type=int) # For early stop. not implemented yet

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    set_seed(args.seed)

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    

    df = pd.read_csv('./data/preprocessed_data.csv')
    shuffled_dataframe = df.sample(frac=1, random_state=42).reset_index(drop=True)

    checkpoint = torch.load("./ckpt/best_model.pt")
    model_state_dict = checkpoint["model_state_dict"]

    cl_model = modeling.Sup_CLModel(args).to(args.device)
    cl_model.load_state_dict(model_state_dict)
    # test_evaluator = utils.ClusteringEvaluator(df['Abstract']values.tolist(), df_test['Field'])

    # 나중엔 test만 visualization 해야할듯?
    sentence_representations = cl_model.encode(shuffled_dataframe['Abstract'].values.tolist(), batch_size = 64)
    rule = {"NLP":0, "VISION":1, "RS":2, "MI":3}
    labels = [rule[label] for label in shuffled_dataframe['Field']]
    # Apply T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(sentence_representations)

    # Visualize the result
    # color result 추가해야함
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels)
    plt.title('T-SNE Visualization')
    plt.legend()
    plt.show()
    plt.savefig("asdf")

main()