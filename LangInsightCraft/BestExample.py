import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class BestExample:
    def __init__(self, csv_file_path: str):
        """Load CSV file and initialize the Sentence Transformer model."""
        self.df = pd.read_csv(csv_file_path)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        """Generates All-mpnet-base-v2 embeddings for the input text."""
        sentence_embedding = self.model.encode(text, convert_to_tensor=True)
        return sentence_embedding

    def find_top_n_similar_descriptions(self, input_description: str, n: int = 10) -> pd.DataFrame:
        """Finds the top N semantically similar functional descriptions."""
        input_embedding = self.get_sentence_embedding(input_description)
        similarities = []

        for index, row in self.df.iterrows():
            description = row['Functional Description']
            description_embedding = self.get_sentence_embedding(description)
            similarity = util.pytorch_cos_sim(input_embedding, description_embedding).item()
            similarities.append((index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_n_indices = [index for index, similarity in similarities[:n]]
        top_n_descriptions = self.df.iloc[top_n_indices]
        
        return top_n_descriptions[['Functional Description', 'Method Name']]
