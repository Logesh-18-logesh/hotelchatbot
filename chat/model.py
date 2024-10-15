import json
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load dataset from Hugging Face
ds = load_dataset("M-A-E/hotel-booking-assistant-raw-chats")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("rajatsingh/autotrain-chatbot-hotel-57870132936")

# Add a padding token explicitly
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Preprocess the dataset
def extract_messages(examples):
    # Concatenate the messages in a single string to use for training
    texts = []
    for messages in examples['messages']:
        # Join the contents of the messages
        text = " ".join([message["content"] for message in messages])
        texts.append(text)
    return {"text": texts}

# Extract messages from the dataset
formatted_ds = ds.map(extract_messages, batched=True)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Tokenize the dataset
tokenized_ds = formatted_ds.map(tokenize_function, batched=True)

# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = encodings['input_ids']  # Set labels to input_ids for next-token prediction

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx]),  # Include labels
        }

    def __len__(self):
        return len(self.input_ids)

# Create the train dataset
train_dataset = CustomDataset(tokenized_ds['train'])

# Initialize the model (keep using GPT2LMHeadModel if the task is next-token prediction)
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Resize the model's token embeddings to account for the added special tokens
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./hotel-booking-model')
tokenizer.save_pretrained('./hotel-booking-model')
