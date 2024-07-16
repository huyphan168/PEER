import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
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

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=0)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)