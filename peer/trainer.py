import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    batch_losses = []
    for batch in tqdm(train_loader, disable=torch.distributed.get_rank() != 0):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        optimizer.zero_grad()
        
        # Shift the input_ids and attention_mask to create targets
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()
        
        outputs = model(input_ids)
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        # Calculate loss (ignore padding token, usually 0)
        loss = F.cross_entropy(outputs, targets, ignore_index=0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_losses.append(loss.item())

    return total_loss / len(train_loader), batch_losses

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    batch_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=0, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += (input_ids != 0).sum().item()
            batch_losses.append(loss.item() / (input_ids != 0).sum().item())
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, batch_losses