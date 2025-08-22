import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd 

en_ds=pd.read_csv(r"C:\Users\g\Desktop\ML\small_vocab_en.csv")
fr_ds=pd.read_csv(r"C:\Users\g\Desktop\ML\small_vocab_fr.csv")

en_ds=en_ds.rename(columns={'new jersey is sometimes quiet during autumn , and it is snowy in april .' : 'txt'})
fr_ds=fr_ds.rename(columns={"new jersey est parfois calme pendant l' automne , et il est neigeux en avril ." : 'txt'})

en_ds=en_ds.values.tolist()
fr_ds=fr_ds.values.tolist()

en_ds=[' '.join(txt) for txt in en_ds]
fr_ds=[' '.join(txt) for txt in fr_ds]


train_size=int(len(en_ds) * 0.999)

train_en=en_ds[:train_size]
test_en=en_ds[train_size:]

train_fr=fr_ds[:train_size]
test_fr=fr_ds[train_size:]


tokenizer= BertTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size=tokenizer.vocab_size


class DS(Dataset):
    def __init__(self, src_sent, trg_sent, src_max_length, trg_max_length, tokenizer):
        super().__init__()
        self.src_sent=src_sent
        self.trg_sent=trg_sent
        self.src_max_length=src_max_length
        self.trg_max_length=trg_max_length
        self.tokenizer=tokenizer

        assert len(src_sent) == len(trg_sent)

    def __len__(self):
        return len(self.src_sent)

    def __getitem__(self, idx):
        src_tokenized=self.tokenizer(self.src_sent[idx], padding='max_length', truncation=True, return_tensors='pt', max_length=self.src_max_length)
        trg_tokenized=self.tokenizer(self.trg_sent[idx], padding='max_length', truncation=True, return_tensors='pt', max_length=self.trg_max_length)

        src_token_ids=src_tokenized['input_ids'].squeeze(0)
        src_padding_mask=src_tokenized['attention_mask'].squeeze(0)

        trg_input_ids=trg_tokenized['input_ids'][..., :-1].squeeze(0)
        trg_loss_ids=trg_tokenized['input_ids'][..., 1:].squeeze(0)
        trg_padding_mask=trg_tokenized['attention_mask'][..., :-1].squeeze(0)
        trg_causal_mask=self.generate_causal_mask(trg_input_ids.size(-1), device='cuda').unsqueeze(0)


        return {
            'src_ids':src_token_ids, 
            'src_mask':torch.tensor(src_padding_mask, dtype=torch.bool), 
            'trg_input_ids':trg_input_ids, 
            'trg_loss_ids':trg_loss_ids, 
            'trg_padding_mask':torch.tensor(trg_padding_mask, dtype=torch.bool), 
            'trg_causal_mask':trg_causal_mask, 
            'reference':self.trg_sent[idx]  

        }
    
    @staticmethod
    def generate_causal_mask(seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
ds_train=DS(train_en, train_fr, 35, 36, tokenizer)
loader_train=DataLoader(ds_train, batch_size=64)

ds_test=DS(test_en, test_fr, 35, 36, tokenizer)
loader_test=DataLoader(ds_test, batch_size=12)