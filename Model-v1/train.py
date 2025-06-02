import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, d_model=768, hidden_size=512):
        super().__init__()
        
        # Shared layers for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Vulnerability detection head (without sigmoid)
        self.vulnerability_head = nn.Linear(hidden_size // 2, 1)
        
        # Synthetic data detection head (without sigmoid)
        self.synthetic_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        vulnerability_logits = self.vulnerability_head(features)
        synthetic_logits = self.synthetic_head(features)
        return vulnerability_logits, synthetic_logits

class SmartContractTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate=0.0001,
        weight_decay=0.01,
        max_grad_norm=1.0,
        pad_token_id=0,
        d_model=768,
        gpu_id=1  # Add GPU ID parameter
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gpu_id = gpu_id
        
        # Initialize discriminator
        self.discriminator = Discriminator(d_model=d_model)
        
        # Initialize loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.bce_loss = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.max_grad_norm = max_grad_norm
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'vulnerability_loss': [],
            'synthetic_loss': []
        }
        
        # Set CUDA device
        torch.cuda.set_device(self.gpu_id)
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self):
        self.model.train()
        self.discriminator.train()
        total_loss = 0
        total_vuln_loss = 0
        total_synth_loss = 0
        batch_count = 0
        
        # Move models to specified GPU
        device = torch.device(f"cuda:{self.gpu_id}")
        self.model = self.model.to(device)
        self.discriminator = self.discriminator.to(device)
        
        progress_bar = tqdm(self.train_dataloader, desc='Training')
        
        for batch in progress_bar:
            try:
                # Clear cache before each batch
                torch.cuda.empty_cache()
                
                # Move all tensors to the same device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                path_input_ids = batch['path_input_ids'].to(device)
                path_attention_mask = batch['path_attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                # Convert label to vulnerability indicator and reshape
                is_vulnerable = batch['label'].float().to(device).view(-1, 1)
                
                # Train discriminator
                self.discriminator_optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # Forward pass through generator
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        path_input_ids=path_input_ids,
                        path_attention_mask=path_attention_mask,
                        target_ids=target_ids
                    )
                    
                    # Get encoder output for discriminator
                    encoder_output = outputs['encoder_output']
                    
                    # Train discriminator on real data
                    vuln_pred, synth_pred = self.discriminator(encoder_output.detach())
                    vuln_loss_real = self.bce_loss(vuln_pred, is_vulnerable)
                    
                    # For synthetic detection, randomly choose between input_ids and decoder output
                    if torch.rand(1) < 0.5:
                        synth_loss_real = self.bce_loss(synth_pred, torch.zeros_like(synth_pred))
                        is_synthetic = torch.zeros_like(synth_pred)
                    else:
                        synth_loss_real = self.bce_loss(synth_pred, torch.ones_like(synth_pred))
                        is_synthetic = torch.ones_like(synth_pred)
                    
                    # Generate synthetic data
                    with torch.no_grad():
                        synthetic_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            path_input_ids=path_input_ids,
                            path_attention_mask=path_attention_mask,
                            target_ids=None
                        )
                        synthetic_encoder_output = synthetic_outputs['encoder_output']
                    
                    # Train discriminator on synthetic data
                    vuln_pred_synth, synth_pred_synth = self.discriminator(synthetic_encoder_output)
                    vuln_loss_synth = self.bce_loss(vuln_pred_synth, torch.zeros_like(vuln_pred_synth))
                    synth_loss_synth = self.bce_loss(synth_pred_synth, torch.ones_like(synth_pred_synth))
                    
                    # Combined discriminator loss
                    d_loss = (vuln_loss_real + synth_loss_real + vuln_loss_synth + synth_loss_synth) / 2
                
                # Backward pass for discriminator with gradient scaling
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.discriminator_optimizer)
                self.scaler.update()
                
                # Train generator
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    logits = outputs['logits']
                    target_ids = target_ids[:, 1:].contiguous().view(-1)
                    gen_loss = self.criterion(logits, target_ids)
                    
                    # Generate new synthetic data for generator training
                    synthetic_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        path_input_ids=path_input_ids,
                        path_attention_mask=path_attention_mask,
                        target_ids=None
                    )
                    synthetic_encoder_output = synthetic_outputs['encoder_output']
                    
                    # Adversarial loss for generator
                    vuln_pred, synth_pred = self.discriminator(synthetic_encoder_output)
                    
                    # Generator losses
                    adv_vuln_loss = self.bce_loss(vuln_pred, torch.zeros_like(vuln_pred))
                    adv_synth_loss = self.bce_loss(synth_pred, torch.zeros_like(synth_pred))
                    diversity_loss = -torch.mean(torch.abs(synthetic_encoder_output - encoder_output.detach()))
                    
                    # Combined generator loss
                    total_loss = gen_loss + 0.1 * (adv_vuln_loss + adv_synth_loss) + 0.05 * diversity_loss
                
                # Backward pass for generator with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                total_loss += gen_loss.item()
                total_vuln_loss += (vuln_loss_real.item() + vuln_loss_synth.item()) / 2
                total_synth_loss += (synth_loss_real.item() + synth_loss_synth.item()) / 2
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'gen_loss': gen_loss.item(),
                    'vuln_loss': (vuln_loss_real.item() + vuln_loss_synth.item()) / 2,
                    'synth_loss': (synth_loss_real.item() + synth_loss_synth.item()) / 2,
                    'diversity_loss': diversity_loss.item(),
                    'is_synthetic': is_synthetic.mean().item()
                })
                
                # Clear some memory
                del outputs, synthetic_outputs, encoder_output, synthetic_encoder_output
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                print(f"Batch keys: {batch.keys()}")
                print(f"Label shape: {batch['label'].shape if 'label' in batch else 'No label'}")
                print(f"Label type: {batch['label'].dtype if 'label' in batch else 'No label'}")
                print(f"Vuln pred shape: {vuln_pred.shape if 'vuln_pred' in locals() else 'Not created'}")
                print(f"Synth pred shape: {synth_pred.shape if 'synth_pred' in locals() else 'Not created'}")
                print(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB allocated")
                print(f"GPU Memory: {torch.cuda.memory_reserved(device) / 1024**2:.2f}MB reserved")
                continue
        
        return {
            'gen_loss': total_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'synth_loss': total_synth_loss / batch_count if batch_count > 0 else float('inf')
        }
    
    def validate(self):
        self.model.eval()
        self.discriminator.eval()
        total_loss = 0
        total_vuln_loss = 0
        total_synth_loss = 0
        batch_count = 0
        
        device = torch.device(f"cuda:{self.gpu_id}")
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Get batch data and move to GPU
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    path_input_ids = batch['path_input_ids'].to(device)
                    path_attention_mask = batch['path_attention_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    # Convert label to vulnerability indicator and reshape
                    is_vulnerable = batch['label'].float().to(device).view(-1, 1)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        path_input_ids=path_input_ids,
                        path_attention_mask=path_attention_mask,
                        target_ids=target_ids
                    )
                    
                    # Calculate generator loss
                    logits = outputs['logits']
                    target_ids = target_ids[:, 1:].contiguous().view(-1)
                    gen_loss = self.criterion(logits, target_ids)
                    
                    # Discriminator predictions
                    vuln_pred, synth_pred = self.discriminator(outputs['encoder_output'])
                    vuln_loss = self.bce_loss(vuln_pred, is_vulnerable)
                    
                    # For synthetic detection, use a mix of real and synthetic data
                    synth_target = torch.zeros_like(synth_pred)
                    synth_loss = self.bce_loss(synth_pred, synth_target)
                    
                    # Update metrics
                    total_loss += gen_loss.item()
                    total_vuln_loss += vuln_loss.item()
                    total_synth_loss += synth_loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    print(f"Batch keys: {batch.keys()}")
                    print(f"Label shape: {batch['label'].shape if 'label' in batch else 'No label'}")
                    print(f"Label type: {batch['label'].dtype if 'label' in batch else 'No label'}")
                    print(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB allocated")
                    print(f"GPU Memory: {torch.cuda.memory_reserved(device) / 1024**2:.2f}MB reserved")
                    continue
        
        return {
            'gen_loss': total_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'synth_loss': total_synth_loss / batch_count if batch_count > 0 else float('inf')
        }
    
    def train(self, num_epochs, checkpoint_dir='checkpoints'):
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['gen_loss'])
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['gen_loss'])
            self.training_history['val_loss'].append(val_metrics['gen_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['vulnerability_loss'].append(train_metrics['vuln_loss'])
            self.training_history['synthetic_loss'].append(train_metrics['synth_loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Vulnerability Loss: {train_metrics['vuln_loss']:.4f}")
            print(f"Synthetic Loss: {train_metrics['synth_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if validation loss improved
            if val_metrics['gen_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['gen_loss']
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                    'val_loss': val_metrics['gen_loss'],
                    'training_history': self.training_history
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save latest checkpoint
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                'val_loss': val_metrics['gen_loss'],
                'training_history': self.training_history
            }, latest_checkpoint_path) 