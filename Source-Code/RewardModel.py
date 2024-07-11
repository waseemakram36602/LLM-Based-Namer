from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

class RewardModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def semantic_similarity(self, am, sm):
        vec_am = self.encode_text(am)
        vec_sm = self.encode_text(sm)
        return cosine_similarity(vec_am, vec_sm)[0][0]

    def edit_distance(self, am, sm):
        return Levenshtein.distance(am, sm)

    def reward_score(self, am, sm, w_sim, w_edit):
        sim_score = self.semantic_similarity(am, sm)
        edit_score = self.edit_distance(am, sm) / max(len(am), len(sm))
        return w_sim * sim_score - w_edit * edit_score
