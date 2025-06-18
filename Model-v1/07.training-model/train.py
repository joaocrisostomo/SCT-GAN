import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SolidityTokenConstraints:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Solidity keywords that must be followed by specific tokens
        self.keyword_constraints = {
            'contract': ['is', '{', 'interface'],
            'function': ['(', 'view', 'pure', 'external', 'public', 'internal', 'private'],
            'if': ['('],
            'for': ['('],
            'while': ['('],
            'do': ['{'],
            'try': ['{'],
            'catch': ['{'],
            'require': ['('],
            'assert': ['('],
            'revert': ['('],
            'emit': ['('],
            'return': [';', '('],
            'break': [';'],
            'continue': [';'],
            'throw': [';'],
            'import': ['"', "'"],
            'pragma': ['solidity'],
            'library': ['{', 'is'],
            'interface': ['{', 'is'],
            'struct': ['{'],
            'enum': ['{'],
            'event': ['('],
            'modifier': ['{', '('],
            'using': ['for'],
            'mapping': ['('],
            'address': ['payable', ';', ',', ')'],
            'uint': [';', ',', ')', '['],
            'int': [';', ',', ')', '['],
            'bool': [';', ',', ')'],
            'string': [';', ',', ')'],
            'bytes': [';', ',', ')', '['],
            'memory': [';', ',', ')'],
            'storage': [';', ',', ')'],
            'calldata': [';', ',', ')'],
            'public': [';', ',', ')', '{'],
            'private': [';', ',', ')', '{'],
            'internal': [';', ',', ')', '{'],
            'external': [';', ',', ')', '{'],
            'view': [';', ',', ')', '{'],
            'pure': [';', ',', ')', '{'],
            'payable': [';', ',', ')', '{'],
            'constant': [';', ',', ')'],
            'immutable': [';', ',', ')'],
            'override': [';', ',', ')', '{'],
            'virtual': [';', ',', ')', '{'],
            'abstract': ['contract', 'interface'],
            'indexed': [',', ')'],
            'anonymous': [';'],
            'unchecked': ['{'],
            'receive': ['(', '{'],
            'fallback': ['(', '{']
        }
        
        # Token pairs that must be balanced
        self.balanced_pairs = [
            ('{', '}'),
            ('(', ')'),
            ('[', ']'),
            ('"', '"'),
            ("'", "'")
        ]
        
        # Tokens that must be followed by a semicolon
        self.semicolon_required = [
            'return', 'break', 'continue', 'throw',
            'require', 'assert', 'revert'
        ]
        
        # Initialize constraint masks
        self._init_constraint_masks()
        
        # Pre-compute token indices for faster lookup
        self._precompute_token_indices()
    
    def _precompute_token_indices(self):
        """Pre-compute token indices for faster lookup"""
        self.keyword_indices = {}
        self.allowed_token_indices = {}
        
        for keyword, allowed_tokens in self.keyword_constraints.items():
            keyword_idx = self.tokenizer.convert_tokens_to_ids(keyword)
            if keyword_idx != self.tokenizer.unk_token_id:
                self.keyword_indices[keyword_idx] = True
                allowed_indices = []
                for token in allowed_tokens:
                    token_idx = self.tokenizer.convert_tokens_to_ids(token)
                    if token_idx != self.tokenizer.unk_token_id:
                        allowed_indices.append(token_idx)
                if allowed_indices:
                    self.allowed_token_indices[keyword_idx] = torch.tensor(allowed_indices)
    
    def _init_constraint_masks(self):
        """Initialize masks for token constraints"""
        self.keyword_mask = torch.zeros(self.tokenizer.vocab_size)
        self.balance_mask = torch.zeros(self.tokenizer.vocab_size)
        self.semicolon_mask = torch.zeros(self.tokenizer.vocab_size)
        
        # Convert token constraints to indices
        for keyword, allowed_tokens in self.keyword_constraints.items():
            keyword_idx = self.tokenizer.convert_tokens_to_ids(keyword)
            if keyword_idx != self.tokenizer.unk_token_id:
                for token in allowed_tokens:
                    token_idx = self.tokenizer.convert_tokens_to_ids(token)
                    if token_idx != self.tokenizer.unk_token_id:
                        self.keyword_mask[token_idx] = 1
    
    def apply_constraints_batch(self, logits, prev_tokens):
        """Apply Solidity-specific constraints to logits for a batch of sequences"""
        batch_size = prev_tokens.size(0)
        device = logits.device
        
        # Initialize balance stacks for each sequence in batch
        balance_stacks = [[] for _ in range(batch_size)]
        
        # Get last tokens for each sequence
        last_tokens = prev_tokens[:, -1]
        
        # Create mask for allowed tokens
        allowed_mask = torch.ones_like(logits)
        
        # Apply keyword constraints
        for i in range(batch_size):
            last_token = last_tokens[i].item()
            if last_token in self.keyword_indices:
                allowed_indices = self.allowed_token_indices.get(last_token, None)
                if allowed_indices is not None:
                    mask = torch.zeros_like(logits[i])
                    mask[allowed_indices] = 1
                    allowed_mask[i] = mask
        
        # Apply balanced pair constraints
        for i in range(batch_size):
            for open_token, close_token in self.balanced_pairs:
                open_idx = self.tokenizer.convert_tokens_to_ids(open_token)
                close_idx = self.tokenizer.convert_tokens_to_ids(close_token)
                
                if open_idx != self.tokenizer.unk_token_id and close_idx != self.tokenizer.unk_token_id:
                    if last_tokens[i].item() == open_idx:
                        balance_stacks[i].append(close_idx)
                    elif last_tokens[i].item() == close_idx and balance_stacks[i]:
                        balance_stacks[i].pop()
            
            # Prevent closing tokens if no matching open token
            if not balance_stacks[i]:
                for _, close_token in self.balanced_pairs:
                    close_idx = self.tokenizer.convert_tokens_to_ids(close_token)
                    if close_idx != self.tokenizer.unk_token_id:
                        allowed_mask[i, close_idx] = 0
        
        # Apply semicolon constraints
        for i in range(batch_size):
            last_token_str = self.tokenizer.decode([last_tokens[i].item()])
            if last_token_str in self.semicolon_required:
                semicolon_idx = self.tokenizer.convert_tokens_to_ids(';')
                if semicolon_idx != self.tokenizer.unk_token_id:
                    logits[i, semicolon_idx] *= 2  # Increase probability of semicolon
        
        # Apply masks to logits
        logits = logits.masked_fill(allowed_mask == 0, float('-inf'))
        
        return logits

class SmartContractTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        tokenizer,
        learning_rate=0.0001,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gpu_id=0,
        d_model=768
    ):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Initialize token constraints
        self.token_constraints = SolidityTokenConstraints(tokenizer)
        
        # Get base model parameters (excluding vulnerability heads)
        base_params = []
        contract_head_params = []
        line_head_params = []
        
        for name, param in self.model.named_parameters():
            if 'contract_vulnerability_head' in name:
                contract_head_params.append(param)
            elif 'line_vulnerability_head' in name:
                line_head_params.append(param)
            else:
                base_params.append(param)
        
        # Initialize optimizer with separate parameter groups and lower learning rates
        self.optimizer = optim.AdamW([
            {'params': base_params, 'lr': learning_rate},
            {'params': contract_head_params, 'lr': learning_rate * 1.5},
            {'params': line_head_params, 'lr': learning_rate * 1.5}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler with longer warmup and slower decay
        num_training_steps = len(train_dataloader) * 400  # 400 epochs
        num_warmup_steps = num_training_steps // 5  # 20% warmup
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.max_grad_norm = max_grad_norm
        
        # Initialize focal loss with adjusted parameters
        self.focal_loss = FocalLoss(
            alpha=0.5,
            gamma=1.5,
            reduction='mean'
        )
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'contract_vuln_loss': [],
            'line_vuln_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, epoch):
        self.model.train()
        
        total_gen_loss = 0
        total_contract_vuln_loss = 0
        total_line_vuln_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            desc=f'Epoch {epoch}',
            bar_format='{l_bar}{bar:10}{r_bar}'
        )
        
        for batch in self.train_dataloader:
            try:
                # Move tensors to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                ast_input_ids = batch['ast_input_ids']
                ast_attention_mask = batch['ast_attention_mask']
                vulnerable_lines = batch['vulnerable_lines']
                contract_vulnerabilities = batch['contract_vulnerabilities']
                token_to_line = batch['token_to_line']
                
                # Forward pass with target_ids for training
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask,
                    target_ids=input_ids,  # Use input_ids as target for training
                    token_to_line=token_to_line
                )
                
                # Get vulnerability predictions
                contract_vuln_logits = outputs['contract_vulnerability_logits']
                line_vuln_logits = outputs['line_vulnerability_logits']
                
                # Calculate contract-level vulnerability loss
                contract_vuln_loss = self.focal_loss(
                    contract_vuln_logits,
                    contract_vulnerabilities.float()
                )
                
                # Calculate line-level vulnerability loss
                line_vuln_loss = self.focal_loss(
                    line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                    vulnerable_lines.view(-1, self.model.num_vulnerability_types).float()
                )
                
                # Calculate generator loss with label smoothing
                logits = outputs['logits']
                target_ids = outputs['target_ids']
                
                # Apply label smoothing
                smoothing = 0.1
                n_classes = logits.size(-1)
                one_hot = torch.zeros_like(logits).scatter(1, target_ids.unsqueeze(1), 1)
                smooth_one_hot = one_hot * (1 - smoothing) + (smoothing / n_classes)
                
                # Calculate generator loss
                gen_loss = -(smooth_one_hot * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                
                # Combined loss with adjusted weights
                total_loss = (
                    gen_loss + 
                    0.2 * contract_vuln_loss +  # Further reduced weight for contract-level detection
                    0.1 * line_vuln_loss       # Further reduced weight for line-level detection
                )
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_gen_loss += gen_loss.item()
                total_contract_vuln_loss += contract_vuln_loss.item()
                total_line_vuln_loss += line_vuln_loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'contract_vuln_loss': f'{contract_vuln_loss.item():.4f}',
                    'line_vuln_loss': f'{line_vuln_loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                print(f"\nError in batch: {str(e)}")
                continue
        
        progress_bar.close()
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'contract_vuln_loss': total_contract_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'line_vuln_loss': total_line_vuln_loss / batch_count if batch_count > 0 else float('inf')
        }

    def validate(self):
        self.model.eval()
        
        total_gen_loss = 0
        total_contract_vuln_loss = 0
        total_line_vuln_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Move tensors to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    ast_input_ids = batch['ast_input_ids']
                    ast_attention_mask = batch['ast_attention_mask']
                    vulnerable_lines = batch['vulnerable_lines']
                    contract_vulnerabilities = batch['contract_vulnerabilities']
                    token_to_line = batch['token_to_line']
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        token_to_line=token_to_line
                    )
                    
                    # Get vulnerability predictions
                    contract_vuln_logits = outputs['contract_vulnerability_logits']
                    line_vuln_logits = outputs['line_vulnerability_logits']
                    
                    # Calculate contract-level vulnerability loss
                    contract_vuln_loss = self.focal_loss(
                        contract_vuln_logits,
                        contract_vulnerabilities.float()
                    )
                    
                    # Calculate line-level vulnerability loss
                    line_vuln_loss = self.focal_loss(
                        line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                        vulnerable_lines.view(-1, self.model.num_vulnerability_types).float()
                    )
                    
                    # Calculate generator loss
                    generated_seq = outputs['generated_sequence']
                    target_seq = input_ids[:, :generated_seq.size(1)]
                    
                    gen_logits = F.one_hot(generated_seq, num_classes=self.model.vocab_size).float()
                    gen_logits = gen_logits.view(-1, self.model.vocab_size)
                    target_seq = target_seq.contiguous().view(-1)
                    
                    gen_loss = F.cross_entropy(gen_logits, target_seq)
                    
                    # Combined loss
                    total_loss = (
                        gen_loss + 
                        0.5 * contract_vuln_loss +
                        0.3 * line_vuln_loss
                    )
                    
                    total_gen_loss += gen_loss.item()
                    total_contract_vuln_loss += contract_vuln_loss.item()
                    total_line_vuln_loss += line_vuln_loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'contract_vuln_loss': total_contract_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'line_vuln_loss': total_line_vuln_loss / batch_count if batch_count > 0 else float('inf')
        }

    def train(self, num_epochs, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['gen_loss'])
            self.training_history['val_loss'].append(val_metrics['gen_loss'])
            self.training_history['contract_vuln_loss'].append(train_metrics['contract_vuln_loss'])
            self.training_history['line_vuln_loss'].append(train_metrics['line_vuln_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Contract Vulnerability Loss: {train_metrics['contract_vuln_loss']:.4f}")
            print(f"Line Vulnerability Loss: {train_metrics['line_vuln_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if validation loss improved
            if val_metrics['gen_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['gen_loss']
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['gen_loss'],
                    'training_history': self.training_history
                }, checkpoint_path)
                print(f"🎉 New best validation loss! Saved checkpoint to {checkpoint_path}")
            
            # Save latest checkpoint
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_metrics['gen_loss'],
                'training_history': self.training_history
            }, latest_checkpoint_path) 
