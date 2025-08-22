from models.transformer import TranslateBatch, Translate, Transformer
from data.trsl_dataset import  vocab_size, tokenizer, loader_test
import torch 
import sacrebleu
import evaluate
rouge = evaluate.load("rouge")
model=Transformer(512, 4, vocab_size, 4)

model.load_state_dict(torch.load("C:\\Users\\g\\Desktop\\ML\\translation_model.pth"))

inference_batch=TranslateBatch(model, 20, tokenizer)

for batch in loader_test:
    src_ids=batch['src_ids'].to('cuda')
    src_padding_mask=batch['src_mask'].to('cuda')
    refs=batch['reference']

    preds=inference_batch(src_ids, src_padding_mask)
    
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    rouge_scores = rouge.compute(predictions=preds, references=refs)

    print(bleu)
    print(rouge_scores)
    print(' ')

inference_single=Translate(model, 20, tokenizer)
print(inference_single('I love Italy'))