"""Training loop from the supplied DVFM implementation."""

import torch
import torch.optim as optim

def train_dvfm(model, train_loader, val_loader, n_epochs=200, lr=1e-3, 
               beta_max=1.0, warmup_epochs=50, free_bits=0.0, device='cpu'):
    """
    Train DVFM with KL annealing
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(n_epochs):
        # Update beta (KL annealing)
        beta = min(beta_max, epoch / warmup_epochs) if warmup_epochs > 0 else beta_max
        
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for x, time, event in train_loader:
            x, time, event = x.to(device), time.to(device), event.to(device)
            
            optimizer.zero_grad()
            shape_T, scale_T, shape_C, scale_C, mu, logvar = model(x, time, event)
            loss, recon, kl = model.loss_function(shape_T, scale_T, shape_C, scale_C,
                                                   mu, logvar, time, event, 
                                                   beta, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, time, event in val_loader:
                x, time, event = x.to(device), time.to(device), event.to(device)
                shape_T, scale_T, shape_C, scale_C, mu, logvar = model(x, time, event)
                loss, _, _ = model.loss_function(shape_T, scale_T, shape_C, scale_C,
                                                  mu, logvar, time, event, 
                                                  beta, free_bits)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Beta: {beta:.3f}, "
                  f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}), "
                  f"Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses
