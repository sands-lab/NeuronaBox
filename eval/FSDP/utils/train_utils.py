import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
g_gigabyte = 1024**3

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run



def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


#test
def calculate_time_stats(log_file_path):
    #only work for 1 epoch!
    with open(log_file_path, 'r') as log_file:
        log_content = log_file.read()
    
    if "middle" in log_content:
        print("err")
        return

    with open(log_file_path, 'r') as log_file:
        times = [float(line.strip()) for line in log_file]
    
    if len(times) == 0:
        print("err")
        return
    

    times.sort()
    

    n = len(times)
    trimmed_times = times[n//3: 2*n//3]
    

    mean_time = np.mean(trimmed_times)
    std_time = np.std(trimmed_times)
    
    stats_result = f"Average of the middle 1/3 batches: {mean_time:.6f} seconds\nStandard deviation of the middle 1/3 batches: {std_time:.6f} \n"
    
    print(stats_result)
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(stats_result)
#test


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    
    
    #test
    time_rank=get_date_of_run()+'RANK_'+("1" if int(os.environ["RANK"]) == 1 else "0")
    log_file_path = "./log_file/log_time_"+ time_rank
    print(log_file_path)
    os.makedirs('./log_file/', exist_ok=True)
    open(log_file_path, 'w').close()
    #test
    
    if sampler:
        sampler.set_epoch(epoch)
    # if rank==0:
    #     inner_pbar = tqdm.tqdm(
    #         range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
    #     )
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
        for batch in train_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            start_time = time.time()
            if int(os.environ["MOD_KERNEL_BYPASS"]) != 1:
                optimizer.zero_grad()
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
            loss = output["loss"]
            loss.backward()
            if int(os.environ["MOD_KERNEL_BYPASS"]) != 1:
                optimizer.step()
            end_time = time.time()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch)
            # if rank==0:
            #     inner_pbar.update(1)

            #test
            time_diff = end_time - start_time
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'{time_diff:.6f}\n')
    prof.export_chrome_trace("./tmp/"+time_rank+".json")
               
    calculate_time_stats(log_file_path)    
    #test
    
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        # inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def setup_model(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer
