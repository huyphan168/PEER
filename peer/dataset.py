from datasets import load_dataset
from torch.utils.data import Dataset

class PileDataset(Dataset):
    def __init__(self, file_path, tokenizer, split='train', max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.data = load_dataset(file_path, "wikitext-103-raw-v1", split=split)
        self.data = self.data.filter(lambda x: len(x['text']) > 0)
        if split == "train":
            self.data = self.data.select(range(0,300000))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

        
