import argparse
import datetime
import copy
import numpy as np
import pandas as pd

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

import modeling
import utils


def train(args, model, train_data, evaluator):
    train_data = utils.PaperDataset(args, train_data, model.tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=args.lr)

    best_v_measure = 0

    for epoch_num in range(args.epochs):
        total_loss_train = 0

        model.train()
        for train_input_ids, train_attention_mask, train_label in tqdm(train_dataloader):
            train_input_ids = train_input_ids.to(args.device)
            train_attention_mask = train_attention_mask.to(args.device)
            train_label = train_label.to(args.device)

            model.zero_grad()

            outputs = model(train_input_ids, train_attention_mask) # getting embeddings of abstract
            loss = model.supervised_cl_loss(outputs, train_label)
            print(loss)

            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # clip_grad_norm_(model.parameters(), args.max_norm)

        # Validation
        print("Validation starts!")
        v_measure = evaluator(model)

        print(f"Epochs: {epoch_num + 1} \
        | Train Loss: {total_loss_train / len(train_data): .3f} \
        | Val v_measure: {v_measure: .3f}")

        if best_v_measure < v_measure:
            best_v_measure = v_measure
            best_model = copy.deepcopy(model)
            torch.save(
                {
                    "model": "best_model",
                    "epoch": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_loss": total_loss_train,
                },
                f"./ckpt/best_model.pt",
            )
    
    return best_model


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

    
    cl_model = modeling.Sup_CLModel(args).to(args.device)

    df = pd.read_csv('./data/preprocessed_data.csv')

    ##### modified #####
    #df = df[:200]
    ##### modified #####
    '''
    df_train -> 6
    df_val -> 2
    df_test -> 2
    '''
    df_train, df_val = np.split(df.sample(frac=1, random_state=args.seed), [int(0.6 * len(df))])
    df_val, df_test = np.split(df_val.sample(frac=1, random_state=args.seed), [int(0.5 * len(df_val))])

    val_evaluator = utils.ClusteringEvaluator(df_val['Abstract'], df_val['Field'])
    test_evaluator = utils.ClusteringEvaluator(df_test['Abstract'], df_test['Field'])

    # Train
    best_model = train(args, cl_model, df_train, val_evaluator)

    # Test
    print("@@@@@@@@@@ Test starts @@@@@@@@@@")
    v_measure = test_evaluator(best_model)
    print(f'Test v_measure: {v_measure: .3f}')



    print("Finished!")


## dynamic padding 구현 안함

if __name__ == "__main__":
    main()