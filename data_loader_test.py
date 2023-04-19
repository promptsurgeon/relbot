import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np
import torch 
from torch.utils.data import Dataset

class SocialMediaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, chunk_size=256, overlap=64):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __len__(self):
        return len(self.data)

    def _chunk_text(self, text):
        chunked_text = []
        for start in range(0, len(text), self.chunk_size - self.overlap):
            end = start + self.chunk_size
            chunked_text.append(text[start:end])
        return chunked_text

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        label = self.data.loc[idx, 'label']
        
        # Chunk the text
        chunked_text = self._chunk_text(text)
        
        # Tokenize and embed each chunk
        embeddings = []
        for chunk in chunked_text:
            inputs = self.tokenizer(chunk, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        inputs = {'input_embeddings': torch.tensor(avg_embedding, dtype=torch.float).unsqueeze(0)}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs






sample_data = {'text': ['This is a sample text for testing the SocialMediaDataset class.', 'Another example text to test the class.'],
               'label': [1, 0]}
sample_df = pd.DataFrame(sample_data)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')



sample_dataset = SocialMediaDataset(sample_df, tokenizer)

for i in range(len(sample_dataset)):
    inputs = sample_dataset[i]
    print(f"Sample {i + 1}:")
    print(f"  Averaged Embeddings: {inputs['input_ids'].shape}")
    print(f"  Label: {inputs['labels']}\n")



