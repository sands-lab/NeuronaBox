import time
import torch
import pickle
import os
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from transformers import BertForQuestionAnswering, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# Function to register hooks on each module
def register_hooks_for_all_modules(model):
    # Hook function to print information
    def forward_hook(module, input, output):
        if hasattr(output, 'grad_fn') and output.requires_grad:
            grad_fn = str(output.grad_fn.__class__.__name__)
        else:
            grad_fn = "None"
        print(f"{module.__class__.__name__}: {grad_fn}")

    # Register the hook on all modules
    for module in model.modules():
        module.register_forward_hook(forward_hook)

# Register hooks

if __name__ == '__main__':
    local_rank = 0
    torch.cuda.set_device(local_rank)
    model_name = 'bert-large-uncased'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

   # Dummy context and questions for demonstration purposes
    contexts = ["The SQuAD dataset is used for training question answering models."] * 32
    questions = ["What is the SQuAD dataset used for?"] * 32

    # Tokenize contexts and questions
    inputs = tokenizer(contexts, questions, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    # Create tensor dataset
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
    batch_size = 32

    # Move the model to the GPU
    model.to(local_rank)

    register_hooks_for_all_modules(model)

    
    for key in inputs:
        inputs[key] = inputs[key].to(local_rank)
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters())
    experiment_results_time = np.zeros(100)
    loss_fn = torch.nn.CrossEntropyLoss()
    for iteration in range(1):
        start_time=time.time()
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Calculate loss
        start_positions = torch.randint(0, 32, (batch_size,)).to(local_rank)
        end_positions = torch.randint(0, 32, (batch_size,)).to(local_rank)
        loss = loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_time = time.time() - start_time
