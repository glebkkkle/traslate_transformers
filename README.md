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
BLEU = 89.71 95.1/93.1/91.3/89.3 (BP = 0.973 ratio = 0.974 hyp_len = 185 ref_len = 190)
{'rouge1': np.float64(0.9843843843843844), 'rouge2': np.float64(0.9759103641456583), 'rougeL': np.float64(0.9851551551551553), 'rougeLsum': np.float64(0.9853103103103104)}
 
BLEU = 83.74 87.7/84.4/82.6/80.4 (BP = 1.000 ratio = 1.053 hyp_len = 179 ref_len = 170)
{'rouge1': np.float64(0.978835978835979), 'rouge2': np.float64(0.9694570135746607), 'rougeL': np.float64(0.978114478114478), 'rougeLsum': np.float64(0.978114478114478)}
 
BLEU = 86.32 91.4/87.7/84.7/81.9 (BP = 1.000 ratio = 1.018 hyp_len = 174 ref_len = 171)
{'rouge1': np.float64(0.9747673782156541), 'rouge2': np.float64(0.9476190476190476), 'rougeL': np.float64(0.9682129173508485), 'rougeLsum': np.float64(0.9689655172413794)}

BLEU = 81.47 88.7/84.6/81.9/78.8 (BP = 0.976 ratio = 0.977 hyp_len = 168 ref_len = 172)
{'rouge1': np.float64(0.9699835796387521), 'rouge2': np.float64(0.9655425219941348), 'rougeL': np.float64(0.9673737373737374), 'rougeLsum': np.float64(0.9673573170124894)}

BLEU = 87.03 91.0/89.0/86.6/83.8 (BP = 0.994 ratio = 0.994 hyp_len = 166 ref_len = 167)
{'rouge1': np.float64(0.9827277998339526), 'rouge2': np.float64(0.9816743827160495), 'rougeL': np.float64(0.9827277998339525), 'rougeLsum': np.float64(0.9827277998339526)}

BLEU = 89.26 91.9/90.0/88.5/86.8 (BP = 1.000 ratio = 1.012 hyp_len = 172 ref_len = 170)
{'rouge1': np.float64(0.9828869047619048), 'rouge2': np.float64(0.9816239316239317), 'rougeL': np.float64(0.9828869047619048), 'rougeLsum': np.float64(0.9828869047619048)}

BLEU = 75.83 84.6/79.0/73.1/67.7 (BP = 1.000 ratio = 1.030 hyp_len = 169 ref_len = 164)
{'rouge1': np.float64(0.9762199699699701), 'rouge2': np.float64(0.9650793650793651), 'rougeL': np.float64(0.9717154654654655), 'rougeLsum': np.float64(0.9717154654654655)}

BLEU = 83.77 87.9/84.5/81.9/81.0 (BP = 1.000 ratio = 1.048 hyp_len = 173 ref_len = 165)
{'rouge1': np.float64(0.9793092212447051), 'rouge2': np.float64(0.9779384501297739), 'rougeL': np.float64(0.9793092212447051), 'rougeLsum': np.float64(0.9793092212447051)}

BLEU = 75.95 82.4/77.8/74.0/70.1 (BP = 1.000 ratio = 1.062 hyp_len = 170 ref_len = 160)
{'rouge1': np.float64(0.9410166630754866), 'rouge2': np.float64(0.8843253968253969), 'rougeL': np.float64(0.9418596925949867), 'rougeLsum': np.float64(0.9419246331011037)}

BLEU = 84.57 88.1/85.4/83.6/81.4 (BP = 1.000 ratio = 1.067 hyp_len = 176 ref_len = 165)
{'rouge1': np.float64(0.9557347670250896), 'rouge2': np.float64(0.9294871794871796), 'rougeL': np.float64(0.9551587301587302), 'rougeLsum': np.float64(0.9551587301587302)}

BLEU = 88.02 89.8/88.4/87.5/86.4 (BP = 1.000 ratio = 1.067 hyp_len = 176 ref_len = 165)
{'rouge1': np.float64(0.9912087912087912), 'rouge2': np.float64(0.9924242424242425), 'rougeL': np.float64(0.9912087912087912), 'rougeLsum': np.float64(0.9912087912087912)}
