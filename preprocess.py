import datasets

dataset = datasets.load_dataset("civil_comments", split="test")

# Print the number of examples in the test split
print(len(dataset))

# Print the first example in the test split
print(dataset[0])

def filter_civil_and_hate_sent(dataset):
    """
    using datasets filter function to filter out civil and hate speech
    """
    
    hate_sent_ds = dataset.filter(lambda example: example['toxicity'] > 0)
    civil_sent_ds = dataset.filter(lambda example: example['toxicity'] == 0)
    return hate_sent_ds, civil_sent_ds

hate_sent_ds, civil_sent_ds = filter_civil_and_hate_sent(dataset)

# print length of hate speech dataset
print(len(hate_sent_ds))

# print length of civil speech dataset
print(len(civil_sent_ds))

# save the hate speech dataset in train.pos file
with open('train.pos', 'w') as f:
    for example in hate_sent_ds:
        f.write(example['text'] + '\n')

# save the civil speech dataset in train.neg file
with open('train.neg', 'w') as f:
    for example in civil_sent_ds:
        f.write(example['text'] + '\n')