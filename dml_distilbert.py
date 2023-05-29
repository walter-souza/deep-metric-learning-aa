import numpy as np
import torch
import transformers

class DMLDistilBert(torch.nn.Module):
    def __init__(self, embedding_size, token_max_length, config="distilbert-base-uncased", use_all_tokens=False, normalize=False):
        super(DMLDistilBert, self).__init__()

        self.name = 'DML_DistilBertV1'
        self.normalize  = normalize
        self.use_all_tokens = use_all_tokens
        
        # Initializing a DistilBERT
        self.model = transformers.DistilBertModel.from_pretrained(config)
        self.configuration = self.model.config

        if self.use_all_tokens:
            self.last_linear = torch.nn.Linear(self.configuration.dim * token_max_length, embedding_size)
        else:
            self.last_linear = torch.nn.Linear(self.configuration.dim, embedding_size)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)

        if self.use_all_tokens:
            out = out.last_hidden_state.view(out.last_hidden_state.shape[0], -1)
        else:
            out = out.last_hidden_state[:,0,:]
        
        out = self.last_linear(out)

        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1)

        return out