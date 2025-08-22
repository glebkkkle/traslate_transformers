import torch 
import numpy as np 
import math 
import torch.nn as nn 


device='cuda'
max_length=35

class Embeddings(torch.nn.Module):
    def __init__(self, embeddings_dim,vocab_size, max_length, device):
        super().__init__()
        self.device=device

        self.static_embeddings=torch.nn.Embedding(vocab_size, embeddings_dim, device=self.device)
        self.pos_embeddings=torch.nn.Embedding(max_length, embeddings_dim, device=self.device)

    def forward(self, tokens):
        positions=torch.arange(0, tokens.size(-1), dtype=torch.long, device=self.device).unsqueeze(0)
        return self.static_embeddings(tokens) + self.pos_embeddings(positions)

class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim, num_heads, device):
        super().__init__()
        assert embeddings_dim % num_heads == 0, "embeddings_dim must be divisible by num_heads"
        self.embeddings_dim = embeddings_dim
        self.num_heads = num_heads
        self.head_dim = embeddings_dim // num_heads

   
        self.W_q = nn.Linear(embeddings_dim, embeddings_dim, device=device)
        self.W_k = nn.Linear(embeddings_dim, embeddings_dim, device=device)
        self.W_v = nn.Linear(embeddings_dim, embeddings_dim, device=device)

        self.W_o = nn.Linear(embeddings_dim, embeddings_dim, device=device)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, Q_input, K_input, V_input, causal_mask=None, padding_mask=None):
        batch_size = Q_input.size(0) 
        seq_len_q = Q_input.size(1) #(batch_size, seq_len_q, embeddings_dim)
        seq_len_k = K_input.size(1)
        seq_len_v = V_input.size(1)

        Q = self.W_q(Q_input).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2) #(batch_size, num_heads, seq_len_q, head_dim)
        K = self.W_k(K_input).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V_input).view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if padding_mask is not None:
            scores=scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if causal_mask is not None:
            scores = scores + causal_mask 

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V) 

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embeddings_dim) #(batch_size, seq_len_q, embeddings_dim)

        output = self.W_o(context)
        return output
    


class FeedForward(torch.nn.Module):
    def __init__(self, input_shape, dropout, device):
        super().__init__()
        self.fc1=torch.nn.Linear(input_shape, input_shape * 4).to(device)
        self.fc2=torch.nn.Linear(input_shape * 4, input_shape).to(device)
        self.dropout=torch.nn.Dropout(p=dropout)


    def forward(self, x):
        out=self.fc1(x)
        out=self.dropout(torch.relu(out))
        out=self.fc2(out)
        return out

class Encoder(nn.Module):
    def __init__(self, embeddings_dim, num_heads, dropout=0.2):
        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.num_heads = num_heads

        self.multihead_layer = MultiHeadAttention(self.embeddings_dim, self.num_heads, device)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(self.embeddings_dim).to(device)

        self.ff_layer = FeedForward(self.embeddings_dim, dropout, device)
        self.ff_dropout = nn.Dropout(dropout)
        self.norm_layer2 = nn.LayerNorm(self.embeddings_dim).to(device)


    def forward(self, embeddings, padding_mask=None):

        identity = embeddings
        out = self.norm_layer1(embeddings)             
        out = self.multihead_layer(out,out,out, padding_mask=padding_mask)      

        out = self.attn_dropout(out)
        out = identity + out                

        identity = out
        out = self.norm_layer2(out)          
        out = self.ff_layer(out)                  
        out = self.ff_dropout(out)
        out = identity + out                    
        return out
    

class Decoder(torch.nn.Module):
    def __init__(self, embeddings_dim, num_heads, device, dropout=0.2):
        super().__init__()
        self.mha=MultiHeadAttention(embeddings_dim, num_heads, device)
        self.attn_dropout=torch.nn.Dropout(dropout)
        self.norm_layer1=torch.nn.LayerNorm(embeddings_dim).to(device)

        self.crossattention=MultiHeadAttention(embeddings_dim, num_heads, device)
        self.crossdropout=torch.nn.Dropout(dropout)
        self.layer_norm2=torch.nn.LayerNorm(embeddings_dim).to(device)

        self.ff_layer=FeedForward(embeddings_dim, dropout, device)
        self.ff_dropout=torch.nn.Dropout(dropout)
        self.layer_norm3=torch.nn.LayerNorm(embeddings_dim).to(device)

    def forward(self, encoder_output, embeddings, causal_mask=None, trg_pad=None):

        out=self.norm_layer1(embeddings)
        out=self.mha(out, out, out, causal_mask, trg_pad)
        out=self.attn_dropout(out) + embeddings
        
        identity=out
        out=self.crossattention(out, encoder_output, encoder_output)
        out=self.layer_norm2(out)
        out=self.crossdropout(out)
        out=identity + out

        identity=out
        out=self.ff_layer(out)
        out=self.layer_norm3(out)
        out=self.ff_dropout(out)
        out= identity + out

        return out

class TransformerEncoder(torch.nn.Module):
    def __init__(self, embeddings_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder(embeddings_dim, num_heads) 
            for _ in range(num_layers)
        ])

    def forward(self, src_embeddings, src_padding_mask=None):
        src_padding_mask=~src_padding_mask if src_padding_mask is not None else None
        for layer in self.layers:
            src_embeddings=layer(src_embeddings, padding_mask=src_padding_mask)
        return src_embeddings

class TransformerDecoder(torch.nn.Module):
    def __init__(self, embeddings_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Decoder(embeddings_dim, num_heads, device) 
            for _ in range(num_layers)
        ])

    def forward(self,encoder_output, trg_embeddings, trg_padding_mask=None, causal_mask=None):
        trg_padding_mask=~trg_padding_mask if trg_padding_mask is not None else None
        for layer in self.layers:
            trg_embeddings=layer(encoder_output, trg_embeddings, causal_mask=causal_mask, trg_pad=trg_padding_mask)
        return trg_embeddings

class Transformer(torch.nn.Module):
    def __init__(self, embeddings_dim, num_heads, vocab_size, num_layers):
        super().__init__()
        self.num_layers=num_layers
        
        self.embeddings_layer_src=Embeddings(embeddings_dim, vocab_size,max_length, device)
        self.embeddings_layer_trg=Embeddings(embeddings_dim, vocab_size,max_length, device)

        self.encoder_layers=TransformerEncoder(embeddings_dim, num_heads, num_layers)
        self.decoder_layers=TransformerDecoder(embeddings_dim, num_heads, num_layers)
        
        self.fc=torch.nn.Linear(embeddings_dim, vocab_size, device=device)

    def forward(self, src_tokens, trg_tokens, src_padding_mask=None, causal_mask=None, trg_padding_mask=None):
        src_embeddings=self.embeddings_layer_src(src_tokens)
        trg_embeddings=self.embeddings_layer_trg(trg_tokens)

        encoder_output=self.encoder_layers(src_embeddings, src_padding_mask)
        decoder_output=self.decoder_layers(encoder_output, trg_embeddings, trg_padding_mask=trg_padding_mask, causal_mask=causal_mask)

        out=self.fc(decoder_output)
        return out

class Translate(torch.nn.Module): # translate single sentence (untokenized)
    def __init__(self, model, max_length, tokenizer):
        super().__init__()
        self.model=model
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.model.eval()

    def forward(self, src_tokens):
        src_token_ids = self.tokenizer(src_tokens, return_tensors='pt')['input_ids'].to('cuda')
        trg_ids = torch.tensor([[self.tokenizer.cls_token_id]], dtype=torch.long, device='cuda')
         
        with torch.no_grad():
            for _ in range(self.max_length):
                out=self.model(src_token_ids, trg_ids)
                
                logits=out[:, -1, :]
            
                next_token_id=torch.argmax(logits, dim=-1, keepdim=True)
                trg_ids=torch.cat([trg_ids, next_token_id], dim=-1)
            
                if next_token_id.item() == self.tokenizer.sep_token_id:
                    break
            
        return self.tokenizer.decode(trg_ids.squeeze(0), skip_special_tokens=True)
    
class TranslateBatch(torch.nn.Module):
    def __init__(self, model, max_length, tokenizer, device='cuda'):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def forward(self, src_ids, src_mask=None):
        """
        src_texts: list of strings, batch of source sentences
        """
        batch_size = src_ids.size(0)

        if src_mask is not None:
            src_mask = torch.tensor(src_mask, dtype=torch.bool, device=self.device)


        trg_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=self.device
        )


        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for _ in range(self.max_length):

                out = self.model(src_ids, trg_ids, src_padding_mask=src_mask)
                logits = out[:, -1, :]  


                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
                trg_ids = torch.cat([trg_ids, next_token_id], dim=-1)

                finished |= (next_token_id.squeeze(-1) == self.tokenizer.sep_token_id)

                if finished.all():
                    break

        decoded = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in trg_ids
        ]

        return decoded