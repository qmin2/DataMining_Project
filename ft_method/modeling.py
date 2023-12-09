import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from typing import Union, List


class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.args = args
        self.model = BertModel.from_pretrained(self.args.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name)
        self.model.requires_grad_(not self.args.freeze_lm)
        self.fcl = nn.Linear(768,  self.args.num_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)  
        embedding = output.last_hidden_state  # ([bsz, seq_len, 768])
        sent_repr = embedding[:, 0]
        linear_output = self.fcl(sent_repr)
        return linear_output
    
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
                outputs = self.model(**features, return_dict=True)
                embeddings = outputs.last_hidden_state[:, 0].detach().cpu() 
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        
        return all_embeddings
