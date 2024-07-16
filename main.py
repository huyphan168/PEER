import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from peer.dataset import PileDataset
from peer.model import PEERLanguageModel
from peer.trainer import train, validate

# main execution
if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Hyperparameters
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    dim = 256
    num_layers = 8
    num_heads = 8
    num_experts = 512 * 512  
    top_k = 16
    batch_size = 6
    num_epochs = 10
    learning_rate = 1e-4
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = PEERLanguageModel(vocab_size, dim, num_layers, num_heads, num_experts, top_k).to(device)
    
    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Load Pile dataset
    train_dataset = PileDataset('Salesforce/wikitext', tokenizer, split='train')
    val_dataset = PileDataset('Salesforce/wikitext', tokenizer, split='validation')
    
    # Use DistributedSampler for the training data
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if local_rank == 0:
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    
    # Training and validation loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"Epoch Training {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        if local_rank == 0:
            print(f"Epoch Validation {epoch+1}/{num_epochs}")
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_peer_language_model.pth')
    
    # Save the final trained model
    if local_rank == 0:
        torch.save(model.state_dict(), 'final_peer_language_model.pth')

    # Clean up
    dist.destroy_process_group()