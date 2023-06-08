import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("tw.csv")

df['products_names_clean'] = df['products_names']
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.()|+*Ｑ．"
df['products_names_clean'] = df['products_names_clean'].str.replace(r"[%s]+"%punc, "").astype(str)
df['products_names_clean'] = df['products_names_clean'].str.replace(r'/', '')
df['products_names_clean'] = df['products_names_clean'].str.replace(' ', '')
df['products_names_clean'] = df['products_names_clean'].str.replace(r'[a-zA-Z0-9]', '')
df['products_names_clean'] = df['products_names_clean'].str.replace(r'[\u0000-\u4DFF]', '') # unicode range for cjk chinese is 4E00-9FFF
df['products_names_clean'].replace('', np.nan, inplace=True)
df.dropna(subset=['products_names_clean'], inplace=True)
df.drop_duplicates(subset=['products_names_clean'], keep="first", inplace=True)

sentences = df['products_names_clean'].tolist()

model_name = 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokens = {'input_ids': [], 'attention_mask': []}

for sentence in sentences:
    new_tokens = tokenizer.encode_plus(sentence, max_length=128, 
                                        truncation=True, padding='max_length', 
                                        return_tensors='pt')
    
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

outputs = model(**tokens)
embeddings = outputs.last_hidden_state
attention = tokens["attention_mask"]
mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
mask_embeddings = embeddings * mask
summed = torch.sum(mask_embeddings, 1)
counts = torch.clamp(mask.sum(1), min=1e-9)
mean_pooled = summed / counts
mean_pooled = mean_pooled.detach().numpy()

all_cos_sims_avgs = []

for j in range(len(mean_pooled)):
    compare_vals = [x for i,x in enumerate(mean_pooled) if i!=j]
    sim = cosine_similarity(
        [mean_pooled[j]],
        compare_vals)
    all_cos_sims_avgs.append(np.mean(sim[0]))

df['cos_sim'] = all_cos_sims_avgs
df.to_csv(f"tw_sims.csv", index=False, encoding='utf_8_sig')