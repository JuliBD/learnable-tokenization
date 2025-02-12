import pandas as pd

class VocabManager:
    def __init__(self,
                 starting_vocab: list[tuple[int]] = []
                 ):
        self.vocab = pd.DataFrame(
            {
                "token_id": starting_vocab
            }
        )
        self.empty_idxs = [] # this will be used for implementing forgetting of vocab
        self.new_vocab = []

    def _add_token(self, token_id):
        self.new_vocab.append(token_id)
        if len(self.empty_idxs) > 0:
            new_token_idx = self.empty_idxs[0]
            self.empty_idxs.pop(0)
            self.vocab.loc[new_token_idx] = [token_id]
            return new_token_idx
        else:
            length = len(self.vocab)
            new_token_idx = length + 1
            self.vocab.loc[new_token_idx] = [token_id]
            return new_token_idx


    def id2idx(self, token_id):
        try:
            return int(self.vocab[self.vocab["token_id"] == token_id].index[0])
        except:
            return self._add_token(token_id)
    
    def ids2idxs(self, token_ids: list[tuple[int]]):
        return [self.id2idx(token_id) for token_id in token_ids]

    def idx2id(self, token_idx):
        try:
            return self.vocab.loc[token_idx]["token_id"]
        except:
            raise ValueError("unassigned token_idx")
    
    def reset_new_vocab(self):
        self.new_vocab = []
    
    def __len__(self):
        return len(self.vocab)
    
    def _repr_html_(self):
        return self.vocab._repr_html_()
    
import torch
from torch import nn
from torch import Tensor, FloatTensor
class DynamicHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 min_out_features: int = 1
                 ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = min_out_features
        
        self.head = nn.Linear(in_features = in_features, out_features=min_out_features)
    
    def forward(self, input: Tensor):
        return self.head.forward(input)
    
    def expand_head(self,
                    new_out_features,
                    ):
        old_out_features = self.head.out_features
        if new_out_features < old_out_features:
            raise ValueError(f"new_out_features ({new_out_features}) has to be larger than old_out_features ({old_out_features})")
        #new_out_features = old_out_features + expand_out_features_by
        new_in_features = self.head.in_features
        new_head = nn.Linear(in_features = new_in_features, out_features = new_out_features, device=next(self.head.parameters()).device)
        with torch.no_grad():
            # copying the old weights to the larger head, to maintain the old predictions
            # note that the new_head predictions will be slightly different, but this shouldn't
            # change the predictions too much because the new weights are small noise
            new_head.weight[:old_out_features] = self.head.weight
        self.in_features = new_head.in_features
        self.out_features = new_head.out_features

from transformers import LlamaConfig

class DynamicVocabHead(DynamicHead):
    def __init__(self,
                 in_features: int,
                 config: LlamaConfig,
                 starting_vocab: list[tuple[int]] = [],
                 min_expansion: int = 100
                 ):
        self.vocab_manger = VocabManager(starting_vocab)
        super().__init__(in_features, len(self.vocab_manger))
        self.min_expansion = min_expansion
        self.features = None # used to save last output
        self.config = config
        self.MASK_ID = -100

    def forward(self,
                input: FloatTensor,
                batched_tokens: list[list[tuple[int]]]
                ):
        targets = []
        for tokens in batched_tokens:
            beTargets = self.vocab_manger.ids2idxs(tokens)
            targets.append(beTargets)
        
        self.targets = self.ignore_padding(torch.tensor(targets))
        
        # if the vocab exceedes out_features expand head
        if len(self.vocab_manger) > self.out_features:
            new_out_features = len(self.vocab_manger) + self.min_expansion
            self.expand_head(new_out_features)
        self.features = super().forward(input)
        return self.features
    
    def ignore_padding(self, targets):
        pad_token_idx = self.vocab_manger.id2idx((self.config.pad_token_id,))
        return torch.where(targets==pad_token_idx, self.MASK_ID, targets)
    
    def loss(self):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # adjust for next token prediction
        # remove first target (as there is no feature/prediction for it)
        # remove last feature (as there is no target for it)
        features = self.features[:,:-1]
        targets = self.targets[:,1:]
        
        targets = targets.to(features.device)
        loss = loss_fn(features.permute(0,2,1), targets).mean()
        return loss