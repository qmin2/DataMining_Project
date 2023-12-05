import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from typing import Union, List

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'first': the fisrt representation of the last hidden state
    'avg': average of the last layers' hidden states at each token.
    """
    def __init__(self, pooler_type = "first"):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["first", "avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, outputs):
        last_hidden = outputs.last_hidden_state
        if self.pooler_type == 'first': # CLS token
            return last_hidden[:, 0]
        # elif self.pooler_type == "avg":
        #     return ((last_hidden * attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / attention_mask.squeeze(1).sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError
        
class Sup_CLModel(nn.Module):
    def __init__(self, args):
        super(Sup_CLModel, self).__init__()
        self.args = args
        self.model = BertModel.from_pretrained(args.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        self.cl_pooler = Pooler(args.pooling)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids = input_ids, attention_mask = attention_mask, return_dict = True)


    def encode(self, sentences, batch_size=64, **kwargs): 
        '''
        - forwarding every sentences to get their embeddings
        - just for inference
        '''
        self.model.to(self.args.device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in tqdm(range(0, len(sentences), batch_size)):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenizer( 
                sentences_batch,
                return_tensors='pt',
                max_length = self.args.max_seq_len, ## to be checked
                padding=True,
                truncation = True
                )
            features.to(self.args.device)

            self.eval()
            with torch.no_grad():
                embeddings = self.model(**features, return_dict=True)
                embeddings = self.cl_pooler(embeddings).detach().cpu()
                # embeddings = [num_senteces, dim]
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings]) # Without this line, model causes: ValueError: only one element tensors can be converted to Python scalars

        return all_embeddings
    
    def _text_length(self, text: Union[List[int], List[List[int]]]): 
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def supervised_cl_loss(self, outputs, labels):
        '''
        Implementation of equation (2) in "Supervised Contrastive Learning, Prannay Khosla.et.al"
        Not exatcly the same as the paper.
        '''

        embeddings = self.cl_pooler(outputs) # [bsz, dim]
        # sim_matrix = torch.matmul(embeddings, embeddings.T) # [bsz, bsz]
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        sim_matrix = cos(embeddings.unsqueeze(1), embeddings.unsqueeze(0)) # [bsz, bsz]

        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T) # [bsz, bsz]

        augmented_sim_matrix = torch.exp(sim_matrix / self.args.temperature)

        ind = torch.eye(labels.size(0)).bool().to(self.args.device)
        #augmented_sim_matrix[ind] = 0 -> in-place operation
        augmented_sim_matrix = augmented_sim_matrix.clone().masked_fill_(ind, 0) # to avoid in-place operation


        total = torch.sum(augmented_sim_matrix, dim=1)
        pos= torch.sum(augmented_sim_matrix * positive_mask, dim=1)
        ### pos의 수로 나눠줘야 하나?
        # pos= torch.sum(augmented_sim_matrix * positive_mask, dim=1) / torch.sum(positive_mask)
        
        loss = torch.log(total + 1e-8) - torch.log(pos+ 1e-8)

        return torch.mean(loss)
    