


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextRelationalGraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, adj_size, activation, 
                 basis_num=2, use_text=True):
        super(TextRelationalGraphAttention, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.adj_size   = adj_size
        self.basis_num  = basis_num
        self.activation = activation.lower()
        self.use_text = use_text

        # trainable parameters
        # attention guids
        self.text_guids  = nn.ParameterList([nn.Parameter(torch.randn(1, 2*hidden_dim))
                                             for i in range(self.adj_size)])
        self.graph_guids_1 = nn.ParameterList([nn.Parameter(torch.randn(1, self.input_dim))
                                               for i in range(self.adj_size)])
        self.graph_guids_2 = nn.ParameterList([nn.Parameter(torch.randn(1, self.input_dim))
                                               for i in range(self.adj_size)])
        self.guid_a = nn.ParameterList([nn.Parameter(torch.randn(1)) 
                                        for i in range(self.adj_size)])
        # basis decomposition
        self.basis_V = nn.Parameter(torch.randn(self.input_dim, self.basis_num, self.output_dim))    
        self.basis_b = nn.ParameterList([nn.Parameter(torch.randn(1, self.basis_num))
                                         for i in range(self.adj_size)])              
        # context
        self.W_1 = nn.Parameter(torch.randn(2*self.hidden_dim, self.input_dim))
        self.W_2 = nn.Parameter(torch.randn(2*self.hidden_dim, self.input_dim))
        # norm
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, tensors):
        # unpack
        entity_embeddings, text_encodings, adj_to_use = tensors[:3]  
        adj_masks = tensors[3:]                          

        # gnn
        fusioned_embeddings = []
        adjacents = []
        for j in range(len(adj_to_use)):
            i = adj_to_use[j]
            # relation text context vector
            att = self.text_guids[i].view(1, 1, -1) * text_encodings  
            att = torch.sum(att, dim=-1, keepdim=True)          
            att = F.softmax(att, dim=1)
            context = torch.sum(att * text_encodings, dim=1)    
            context_1 = torch.mm(context, self.W_1)              
            context_2 = torch.mm(context, self.W_2)
            guid_a = torch.sigmoid(self.guid_a[i])

            if self.use_text:
                graph_guid_1 = guid_a * self.graph_guids_1[i] + (1-guid_a) * context_1  
                graph_guid_2 = guid_a * self.graph_guids_2[i] + (1-guid_a) * context_2
            else:
                graph_guid_1 = self.graph_guids_1[i]
                graph_guid_2 = self.graph_guids_2[i]

            # text-based attention adjacent
            att_entity_embeddings_1 = torch.mm(entity_embeddings, graph_guid_1.t())  
            att_entity_embeddings_2 = torch.mm(entity_embeddings, graph_guid_2.t())
            A = att_entity_embeddings_1 + att_entity_embeddings_2.t()
            mask = adj_masks[j].to_dense()
            neg_mat = -1e10 * torch.ones_like(mask)
            A = torch.where(mask == 1, A, neg_mat)
            A = F.softmax(A, dim=-1)
            adjacents.append(A) 
            
            fusioned_embedding = torch.mm(A, entity_embeddings)       
            fusioned_embeddings.append(fusioned_embedding)         
            
        fusioned_embeddings = torch.cat(fusioned_embeddings, dim=-1)  

        # feed forwad with basis decomposition
        basis_b = torch.cat([self.basis_b[i] for i in adj_to_use], dim=0)
        W = torch.matmul(basis_b, self.basis_V)   
        W = W.view(len(adj_to_use) * self.input_dim, self.output_dim) 
        fusioned_embedding = torch.mm(fusioned_embeddings, W)     
        fusioned_embedding = self.norm(fusioned_embedding)
        # acitvation
        if self.activation == "relu":
            fusioned_embedding = F.relu(fusioned_embedding)
        elif self.acitvation == "tanh":
            fusioned_embedding = F.tanh(fusioned_embedding)
        elif self.acitvation == "sigmoid":
            fusioned_embedding = F.tanh(fusioned_embedding)

        return fusioned_embedding

if __name__ == '__main__':
    pass