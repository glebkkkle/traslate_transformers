import torch 
from trsl_dataset import loader_train, vocab_size, tokenizer
from encoder_transformers import Transformer, Translate
from transformers import get_scheduler 


EMBEDDINGS_DIM=512
NUM_HEADS=4
NUM_LAYERS=4
VOCAB_SIZE=vocab_size

model=Transformer(EMBEDDINGS_DIM, NUM_HEADS,VOCAB_SIZE, NUM_LAYERS)
loss_fn=torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer=torch.optim.AdamW(model.parameters(), lr=5e-5)


def fit(loader, model, num_epochs, optimizer, loss_fn, vocab_size):
    model.train()
    num_training_steps = num_epochs * len(loader)
    num_warmup_steps = int(0.1 * num_training_steps) 

    scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        for x in loader:
            src_ids=x['src_ids'].to('cuda')
            trg_input_ids=x['trg_input_ids'].to('cuda')
            src_padding_mask=x['src_mask'].to('cuda')
            trg_padding_mask=x['trg_padding_mask'].to('cuda')
            trg_causal_mask=x['trg_causal_mask'].to('cuda')
            trg_loss_ids=x['trg_loss_ids'].to('cuda')

            preds=model(src_ids, trg_input_ids, src_padding_mask, trg_causal_mask, trg_padding_mask)
            
            preds=preds.view(-1, vocab_size)
            trg_loss_ids=trg_loss_ids.view(-1)

            loss=loss_fn(preds, trg_loss_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


fit(loader_train, model, 8, optimizer, loss_fn, VOCAB_SIZE)
torch.save(model.state_dict(), "translation_model.pth")
