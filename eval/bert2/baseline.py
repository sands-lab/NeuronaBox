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

if __name__ == '__main__':

     # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank=int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    local_size=int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank=int(os.environ['GROUP_RANK'])
    
    if global_rank == 0:
        os.environ["MOD_KERNEL_BYPASS"] = "1"
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)
    os.environ["OMPI_COMM_WORLD_RANK"] = str(global_rank)
        
    print("local_rank: {}, global_rank: {}, world_size: {}, local_size: {}, group_rank: {}".format(local_rank, 
                                                                                                   global_rank, world_size, local_size, group_rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",world_size=world_size,rank=global_rank)
    print("Process Group Initialized")
    
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

    print("Dataset created")
    
    # Move the model to the GPU
    model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model,bucket_cap_mb=25,static_graph=False,find_unused_parameters=False)
    
    print("DDP model created")
    
    for key in inputs:
        inputs[key] = inputs[key].to(local_rank)
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters())
    experiment_results_time = np.zeros(100)
    file_name1 = "~/emulator/eval/bert2/bert_baseline"+str(local_size)+"_"+str(world_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print("Starting training")
    
    for iteration in range(100):
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
        print("Iteration: {}, Time: {}".format(iteration, total_time))
        if (global_rank==0) :
            experiment_results_time[iteration] = total_time
    if global_rank == 0:
        np.save(file_name1, experiment_results_time)
    dist.destroy_process_group()