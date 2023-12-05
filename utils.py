import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
import sklearn
import sklearn.cluster
import logging
import torch
import os

logger = logging.getLogger(__name__)

class PaperDataset(Dataset):
    def __init__(self, args, df, tokenizer):

        use_cache = True \
            if os.path.isfile("./data/labels.npy") and os.path.isfile("./data/input_ids.npy") and os.path.isfile("./data/attention_mask.npy") else False
        # use_cache = False

        if use_cache:
            print("Using cached file")
            self.labels = torch.tensor(np.load("./data/labels.npy"))
            self.input_ids = torch.tensor(np.load("./data/input_ids.npy"))
            self.attention_mask = torch.tensor(np.load("./data/attention_mask.npy"))
            self.instances = {'input_ids':self.input_ids, 'attention_mask':self.attention_mask}

        else:
            rule = {"NLP":0, "VISION":1, "RS":2, "MI":3}
            self.labels = torch.tensor([rule[label] for label in df['Field']])

            self.instances = []
            print(("Tokenizing process, it takes about a minute"))
            self.instances = tokenizer(df['Abstract'].values.tolist(),
                                padding='max_length',
                                max_length=args.max_seq_len,
                                truncation=True,
                                return_tensors="pt")

            print("Save cache file")
            np.save("./data/labels.npy", np.array(self.labels))
            np.save("./data/input_ids.npy", np.array(self.instances['input_ids']), allow_pickle=True)
            np.save("./data/attention_mask.npy", np.array(self.instances['attention_mask']), allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.instances['input_ids'][idx], self.instances['attention_mask'][idx], self.labels[idx]

def preprocess():
    # Exclude papers that include 'multi-modal' keywords
    # 이상한 input, ex) abstract에 내용없는것들 잘라야함
    pass

class ClusteringEvaluator():
    def __init__(self, sentences, labels, clustering_batch_size=500, batch_size=32, limit=None):
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences.values.tolist()
        self.labels = labels.values.tolist()
        self.clustering_batch_size = clustering_batch_size
        self.batch_size = batch_size

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        corpus_embeddings = np.asarray(model.encode(self.sentences, batch_size=self.batch_size))

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size, n_init="auto"
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return v_measure