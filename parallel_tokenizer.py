import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')
    # Your code here


random_seed = 42
torch.manual_seed(random_seed)

# Load the dataset
dataset = load_dataset("yelp_review_full")

# Create a custom dataset class
class YelpDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create the dataset and data loader
train_dataset = YelpDataset(dataset["train"], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

# Calculate the size of each part
num_parts = 5
part_size = len(train_dataset) // num_parts

# Process and save each part separately
for i in range(num_parts):
    print(f"Processing Part {i + 1}")
    
    # Define the start and end indices for the current part
    start_index = i * part_size
    end_index = start_index + part_size if i < num_parts - 1 else len(train_dataset)
    
    # Create a subset of the dataset for the current part
    part_dataset = torch.utils.data.Subset(train_dataset, list(range(start_index, end_index)))
    
    # Create a data loader for the current part
    part_loader = DataLoader(part_dataset, batch_size=32, shuffle=True, num_workers=8)
    
    # Process and save the data for the current part
    part_data = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for batch in tqdm(part_loader, desc=f"Processing Part {i + 1}"):
        part_data["input_ids"].append(batch["input_ids"])
        part_data["token_type_ids"].append(batch["token_type_ids"])
        part_data["attention_mask"].append(batch["attention_mask"])
        part_data["labels"].append(batch["labels"])

    # Convert to tensors
    part_data["input_ids"] = torch.cat(part_data["input_ids"], dim=0)
    part_data["token_type_ids"] = torch.cat(part_data["token_type_ids"], dim=0)
    part_data["attention_mask"] = torch.cat(part_data["attention_mask"], dim=0)
    part_data["labels"] = torch.cat(part_data["labels"], dim=0)

    # Save the data for the current part
    torch.save(part_data, f"yelp_data_part_{i + 1}.pt")