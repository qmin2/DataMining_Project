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

import cl_method.modeling
import ft_method.modeling
import cl_method.utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--pooling', default='first', type=str)

    # Hyper parameter
    parser.add_argument('--method', default = "unsup", choices=['unsup', 'sup', 'ft'], type=str)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--freeze_lm', default=False, action='store_true')
    parser.add_argument('--num_classes', default=4, type=int) 

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    set_seed(args.seed)

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    

    df_test = pd.read_csv('./data/df_test.csv')

    checkpoint = torch.load(f"./ckpt/{args.method}_best_model.pt")
    model_state_dict = checkpoint["model_state_dict"]

    if args.method in ['sup', 'unsup']:
        model = cl_method.modeling.CL_Model(args).to(args.device)
    elif args.method == 'ft':
        model = ft_method.modeling.BertClassifier(args).to(args.device)
    elif args.method == 'bert':
        model = cl_method.modeling.CL_Model(args).to(args.device)

    model.load_state_dict(model_state_dict)
    sentence_representations = model.encode(df_test['Abstract'].values.tolist(), batch_size = 64)
    
    rule = {"NLP":0, "VISION":1, "RS":2, "MI":3}
    numeric_labels = [rule[label] for label in df_test['Field']]

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    sr_tsne = tsne.fit_transform(sentence_representations)

    plt.figure(figsize=(10, 8))

    # Scatter plot each class with a different color
    for class_name, numeric_label in rule.items():
        indices = np.array(numeric_labels) == numeric_label
        plt.scatter(sr_tsne[indices, 0], sr_tsne[indices, 1], label=f'{class_name}')


    plt.title(f"{args.method} T-SNE")
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/{args.method}_fig")


if __name__ == "__main__":
    main()
