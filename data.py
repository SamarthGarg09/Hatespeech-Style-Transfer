import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split
# from transformers import DataCollatorWithPadding

def create_tokenizer():
    tokenizer = Tokenizer.from_file('/Data/deeksha/disha/code_p/style_transformer_repl/_transformers/tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=512)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

# STEP1: READ THE FILE AND TOKENIZE THE POSITIVE AND NEGATIVE SENTENCES
class HSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.pos_data, self.neg_data = self.read_data()
        self.min_length = min(len(self.pos_data), len(self.neg_data))
        self.vocab = tokenizer.get_vocab()
    
    def read_data(self):
        with open(self.data_dir +'/'+ self.split + '.pos', 'r') as f:
            pos_data = f.readlines()
        with open(self.data_dir + '/'+self.split + '.neg', 'r') as f:
            neg_data = f.readlines()
        min_length = min(len(pos_data), len(neg_data))
        return pos_data[:min_length], neg_data[:min_length]
    
    def __len__(self):
        return self.min_length
    # Note: No style tokens has been added at the start of the sentence.
    # STEP3: BUILD A VOCABULARY 
    
    # STEP4: NUMERICALIZE THE INPUT SENTENCE (shape->(BATCH_SIZE, INPUT_IDS))
    def numericalize(self, pos_text, neg_text): 
        pos_tok_text = self.tokenizer(pos_text, truncation=True, return_tensors='pt')
        neg_tok_text = self.tokenizer(neg_text, truncation=True, return_tensors='pt')  
        return {
            'pos_input_ids': pos_tok_text['input_ids'],
            'neg_input_ids': neg_tok_text['input_ids'],
            'pos_attention_mask': pos_tok_text['attention_mask'],
            'neg_attention_mask': neg_tok_text['attention_mask'],
        }
    
    def __getitem__(self, idx):
        # return self.numericalize(self.pos_data[idx], self.neg_data[idx])
        return{
            'pos_data': self.pos_data[idx],
            'neg_data': self.neg_data[idx]
        }
# STEP2: SPLIT THE DATA INTO TRAIN, DEV AND TEST SETS
def split_data(data_dir, tokenizer):
    train_dataset = HSDataset(data_dir, tokenizer, split='train')
    test_dataset = HSDataset(data_dir, tokenizer, split='test')
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=0.2)
    return train_dataset, dev_dataset, test_dataset

tokenizer = create_tokenizer()
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    positive_data, negative_data = [], []
    for item in batch:
        positive_data.append(item['pos_data'])
        negative_data.append(item['neg_data'])
    positive_data = tokenizer(positive_data, padding=True, truncation=True, return_tensors='pt')
    positive_mask = positive_data['attention_mask']
    positive_ids = positive_data['input_ids']
    negative_data = tokenizer(negative_data, padding=True, truncation=True, return_tensors='pt')
    negative_mask = negative_data['attention_mask']
    negative_ids = negative_data['input_ids']

    pos_input_ids = pad_sequence(positive_ids, batch_first=True, padding_value=1.0)
    pos_attention_mask = pad_sequence(positive_mask, batch_first=True, padding_value=0.0)
    neg_input_ids = pad_sequence(negative_ids, batch_first=True, padding_value=1.0)
    neg_attention_mask = pad_sequence(negative_mask, batch_first=True, padding_value=0.0)
    return {
        'pos_input_ids': pos_input_ids,
        'pos_attention_mask': pos_attention_mask,
        'neg_input_ids': neg_input_ids,
        'neg_attention_mask': neg_attention_mask
    }

train_dataset, dev_dataset, test_dataset = split_data("/Data/deeksha/disha/code_p/style-transformer/data/yelp", tokenizer)
print("Data split into train, dev and test sets")
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

for batch in train_dataloader:
    break
print(batch['pos_input_ids'].shape)
