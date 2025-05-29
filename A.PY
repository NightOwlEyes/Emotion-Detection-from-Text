from transformers import AutoModel, AutoTokenizer
import torch

model= AutoModel.from_pretrained('uitnlp/CafeBERT')
tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')

encoding = tokenizer('Cà phê được trồng nhiều ở khu vực Tây Nguyên của Việt Nam.', return_tensors='pt')

with torch.no_grad():
  output = model(**encoding)
  print(output)

''' OUTPUT:
C:\Users\GIGABYTE G5\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\huggingface_hub\file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Some weights of the model checkpoint at uitnlp/CafeBERT were not used when initializing XLMRobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.bias', 'lm_head.dense.weight']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLMRobertaModel were not initialized from the model checkpoint at uitnlp/CafeBERT and are newly initialized: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference. 
BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[-0.1781, -0.2744, -0.5208,  ...,  1.0428, -0.5325, -0.2538],
         [ 0.7800,  0.5334,  0.3101,  ...,  1.1459, -0.5468, -0.1908],
         [ 0.1740,  0.2274, -0.3997,  ...,  1.8262, -1.2452,  0.6767],
         ...,
         [ 0.5360,  0.5725, -0.1375,  ...,  1.4096,  0.1334, -1.2914],
         [-0.0195,  0.2803, -0.1538,  ...,  0.6658, -0.3880, -0.4726],
         [-0.0195,  0.2803, -0.1538,  ...,  0.6658, -0.3880, -0.4725]]]), pooler_output=tensor([[-0.3602,  0.3079, -0.6264,  ..., -0.0285,  0.3275,  0.4934]]), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
'''