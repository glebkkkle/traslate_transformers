📝 Transformer-based English → French Translation From Scratch

This project implements a Transformer architecture from scratch for machine translation (English → French). The model is trained on a custom dataset with over 1 million rows, demonstrating state-of-the-art performance with strong translation quality.

🚀 Key Features

  Custom Transformer Implementation
  
  Built from scratch using PyTorch (no high-level libraries like Hugging Face for the model itself).
  
  Encoder–Decoder architecture with multi-head attention, positional encoding, and feed-forward networks.
  
  Tokenizer
  
  Used Hugging Face’s bert-base-uncased tokenizer for text preprocessing.
  
  Includes <SOS>, <EOS>, <PAD>, and <UNK> tokens for handling sequences.
  
  Training
  
  Trained on a custom parallel English–French dataset with 1M+ rows.
  
  Cross-entropy loss with padding mask applied.
  
  Trained on GPU for efficiency.
  
  Evaluation
  
  Implemented BLEU (via sacrebleu) and ROUGE (via rouge-score) metrics.
  
  Achieved high BLEU and ROUGE scores, showing strong alignment with reference translations.
  
  Inference
  
  Batch inference supported via the TranslateBatch class.
  
  Outputs fully decoded token sequences (no need for post-processing).


Evaluation Metrics:

BLEU = 88.02 89.8/88.4/87.5/86.4 (BP = 1.000 ratio = 1.067 hyp_len = 176 ref_len = 165)
{'rouge1': np.float64(0.9912087912087912), 'rouge2': np.float64(0.9924242424242425), 'rougeL': np.float64(0.9912087912087912), 'rougeLsum': np.float64(0.9912087912087912)}

These results indicate highly accurate and fluent translations
