import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer
import math
import random
from data_augmentation import SmartContractAugmenter

# GAN Components
class PathAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Apply self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_output)

class GrammarConstraint(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.grammar_embedding = nn.Embedding(vocab_size, d_model)
        self.grammar_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Apply grammar-aware projection
        return self.grammar_projection(x)

# Note: Discriminator is now integrated into the main model to avoid backward graph issues

class AugmentedContractDataset(Dataset):
    def __init__(self, original_contracts, augmenter, num_variants_per_contract=3):
        self.original_contracts = original_contracts
        self.augmenter = augmenter
        self.num_variants = num_variants_per_contract
        
        # Create training pairs from augmented contracts
        self.training_pairs = []
        for contract in original_contracts:
            pairs = self.augmenter.augment_contract(contract, num_variants_per_contract)
            self.training_pairs.extend(pairs)
        
        print(f"Created {len(self.training_pairs)} training pairs from {len(original_contracts)} contracts")
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        input_contract, target_contract = self.training_pairs[idx]
        
        # Tokenize input and target
        input_encoding = self.augmenter.tokenizer(
            input_contract,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.augmenter.tokenizer(
            target_contract,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create dummy AST data
        ast_input_ids = input_encoding['input_ids'].clone()
        ast_attention_mask = input_encoding['attention_mask'].clone()
        
        # Create dummy vulnerability data
        vulnerable_lines = torch.zeros((1, 1024, 8))  # 8 vulnerability types
        contract_vulnerabilities = torch.zeros((1, 8))
        token_to_line = torch.zeros((1024,), dtype=torch.long)
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'ast_input_ids': ast_input_ids.squeeze(0),
            'ast_attention_mask': ast_attention_mask.squeeze(0),
            'target_ids': target_encoding['input_ids'].squeeze(0),
            'vulnerable_lines': vulnerable_lines.squeeze(0),
            'contract_vulnerabilities': contract_vulnerabilities.squeeze(0),
            'token_to_line': token_to_line
        }

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

class SpatialAwareFocalLoss(nn.Module):
    """
    Focal loss with spatial context awareness for line-level vulnerability detection.
    Considers the spatial relationship between tokens and their vulnerability patterns.
    """
    def __init__(self, alpha=0.01, gamma=4.0, spatial_weight=0.3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spatial_weight = spatial_weight
        self.reduction = reduction
        
    def forward(self, pred, target, token_to_line=None):
        # Standard focal loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # Add spatial context penalty if token_to_line is provided
        if token_to_line is not None and self.spatial_weight > 0:
            spatial_penalty = self._compute_spatial_penalty(pred, target, token_to_line)
            focal_loss = focal_loss + self.spatial_weight * spatial_penalty
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
    def _compute_spatial_penalty(self, pred, target, token_to_line):
        """
        Compute spatial penalty based on vulnerability patterns in nearby lines.
        """
        # Reshape back to batch format for spatial computation
        batch_size = pred.shape[0] // 1024 if pred.shape[0] > 1024 else 1
        seq_len = 1024
        num_classes = pred.shape[1]
        device = pred.device
        
        # Reshape tensors to [batch_size, seq_len, num_classes]
        pred_reshaped = pred.view(batch_size, seq_len, num_classes)
        target_reshaped = target.view(batch_size, seq_len, num_classes)
        token_to_line_reshaped = token_to_line.view(batch_size, seq_len)
        
        # Create spatial penalty tensor
        spatial_penalty = torch.zeros_like(pred_reshaped)
        
        for b in range(batch_size):
            for i in range(seq_len):
                current_line = token_to_line_reshaped[b, i].item()
                
                # Find tokens from nearby lines (within ±2 lines)
                nearby_mask = torch.abs(token_to_line_reshaped[b] - current_line) <= 2
                nearby_mask[i] = False  # Exclude current token
                
                if nearby_mask.any():
                    # Get vulnerability patterns from nearby tokens
                    nearby_targets = target_reshaped[b, nearby_mask]  # [num_nearby, num_classes]
                    nearby_preds = pred_reshaped[b, nearby_mask]      # [num_nearby, num_classes]
                    
                    # If nearby tokens have vulnerabilities, increase penalty for current token
                    if nearby_targets.sum() > 0:
                        # Increase penalty for current token if it should also be vulnerable
                        vulnerability_similarity = torch.sigmoid(nearby_preds).mean(dim=0)  # [num_classes]
                        spatial_penalty[b, i] = vulnerability_similarity * 0.1
        
        # Flatten back to match input shape
        return spatial_penalty.view(-1, num_classes)

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
        learning_rate=1e-6,
        weight_decay=0.1,
        max_grad_norm=1.0,
        gpu_id=0,
        d_model=768,
        use_augmentation=False,
        use_gan=False
    ):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation
        self.use_gan = use_gan
        
        # Store original dataloaders
        self.original_train_dataloader = train_dataloader
        self.original_val_dataloader = val_dataloader
        
        # Use original dataloaders (augmentation will be applied during training)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Initialize token constraints
        self.token_constraints = SolidityTokenConstraints(tokenizer)
        
        # Get base model parameters (excluding vulnerability heads)
        base_params = []
        contract_head_params = []
        line_head_params = []
        discriminator_params = []
        
        for name, param in self.model.named_parameters():
            if 'disc_' in name and self.use_gan:
                discriminator_params.append(param)
            elif 'contract_vulnerability_head' in name:
                contract_head_params.append(param)
            elif 'line_vulnerability_head' in name or 'spatial_attention' in name:
                line_head_params.append(param)
            else:
                base_params.append(param)
        
        # Initialize optimizer with different learning rates for different components
        param_groups = [
            {'params': base_params, 'lr': learning_rate},
            {'params': contract_head_params, 'lr': learning_rate * 2.0},  # Higher LR for vulnerability heads
            {'params': line_head_params, 'lr': learning_rate * 8.0},  # Much higher LR for line vulnerability heads
        ]
        
        # Add discriminator parameters if GAN is enabled
        if self.use_gan and discriminator_params:
            param_groups.append({'params': discriminator_params, 'lr': learning_rate * 0.5})  # Lower LR for discriminator
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,  
            patience=5,   
            verbose=True,
            min_lr=1e-6   # Increased from 1e-7 to prevent too small LR
        )
        
        self.max_grad_norm = max_grad_norm
        
        # IMPROVED: Loss functions with better handling of extreme class imbalance
        self.focal_loss = FocalLoss(
            alpha=0.25,  # Match working version
            gamma=2.0,   # Match working version
            reduction='mean'
        )
        
        # NEW: Spatial-aware focal loss for line vulnerabilities
        self.spatial_focal_loss = SpatialAwareFocalLoss(
            alpha=0.01,  # Very low alpha for extreme class imbalance
            gamma=4.0,   # Higher gamma for more aggressive down-weighting
            spatial_weight=0.3,  # Weight for spatial context
            reduction='mean'
        )
        
        self.generator_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 24  # Increased from 10 to give more time for improvement
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'contract_vuln_loss': [],
            'line_vuln_loss': [],
            'learning_rate': [],
            'discriminator_loss': []
        }
        
        # Verify learning rate is set correctly
        print(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']}")
        if self.optimizer.param_groups[0]['lr'] > 1e-4:  # Reduced from 1e-3
            print("WARNING: Learning rate is too high! Setting to 1e-4")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-4

        # Initialize augmenter if needed
        if use_augmentation:
            self.augmenter = SmartContractAugmenter(tokenizer.name_or_path)
            print("Data augmentation enabled")
        else:
            print("Using original training data")
        
        # GAN loss function
        if use_gan:
            self.adversarial_loss = nn.BCEWithLogitsLoss()
            print("GAN training enabled - discriminator integrated into model")
            # Verify model has GAN enabled
            if hasattr(self.model, 'use_gan') and self.model.use_gan:
                print("✓ Model has GAN discriminator components")
            else:
                print("✗ WARNING: Model does not have GAN discriminator components!")
                print("Make sure to initialize the model with use_gan=True")
        else:
            print("GAN training disabled")
        
        # Analyze dataset vulnerability distribution
        print("\n=== Dataset Vulnerability Analysis ===")
        total_contracts = 0
        total_contract_vulns = 0
        total_line_vulns = 0
        
        for batch in self.train_dataloader:
            batch_size = batch['contract_vulnerabilities'].size(0)
            total_contracts += batch_size
            
            # Count contract-level vulnerabilities
            contract_vulns = batch['contract_vulnerabilities'].sum().item()
            total_contract_vulns += contract_vulns
            
            # Count line-level vulnerabilities
            line_vulns = batch['vulnerable_lines'].sum().item()
            total_line_vulns += line_vulns
        
        contract_vuln_rate = total_contract_vulns / total_contracts if total_contracts > 0 else 0
        line_vuln_rate = total_line_vulns / (total_contracts * 1024 * 8) if total_contracts > 0 else 0  # 1024 seq_len * 8 vuln_types
        
        print(f"Total contracts: {total_contracts}")
        print(f"Total contract vulnerabilities: {total_contract_vulns}")
        print(f"Total line vulnerabilities: {total_line_vulns}")
        print(f"Contract vulnerability rate: {contract_vuln_rate:.4f} ({contract_vuln_rate*100:.2f}%)")
        print(f"Line vulnerability rate: {line_vuln_rate:.6f} ({line_vuln_rate*100:.4f}%)")
        
        # Store vulnerability rates for dynamic weighting
        self.contract_vuln_rate = contract_vuln_rate
        self.line_vuln_rate = line_vuln_rate
        
        # IMPROVED: Dynamic loss weights based on vulnerability rates
        if line_vuln_rate < 0.001:  # Extreme imbalance
            self.line_vuln_weight = 1000.0  # Much higher weight for line vulnerabilities
            print(f"⚠️  Extreme line vulnerability imbalance detected. Using weight: {self.line_vuln_weight}")
        elif line_vuln_rate < 0.01:  # High imbalance
            self.line_vuln_weight = 500.0
            print(f"⚠️  High line vulnerability imbalance detected. Using weight: {self.line_vuln_weight}")
        else:
            self.line_vuln_weight = 200.0
            print(f"Line vulnerability weight: {self.line_vuln_weight}")

    def _create_augmented_batch(self, batch):
        """Create augmented training pairs from a batch"""
        if not self.use_augmentation:
            return batch
        
        # Check if source_code is available
        if 'source_code' not in batch:
            print("Warning: source_code not found in batch. Skipping augmentation.")
            return batch
        
        # Extract source codes and vulnerability data from batch
        source_codes = batch['source_code']
        contract_names = batch['contract_name']
        original_contract_vulns = batch['contract_vulnerabilities']
        original_vulnerable_lines = batch['vulnerable_lines']
        original_token_to_line = batch['token_to_line']
        
        # Create augmented pairs
        augmented_pairs = []
        for source_code, contract_name in zip(source_codes, contract_names):
            num_variants = random.randint(2, 3)
            pairs = self.augmenter.augment_contract(source_code, num_variants)
            augmented_pairs.extend(pairs)
        
        # Randomly sample pairs to maintain batch size
        if len(augmented_pairs) > len(source_codes):
            selected_pairs = random.sample(augmented_pairs, len(source_codes))
        else:
            selected_pairs = augmented_pairs
        
        # Create new batch with augmented data
        new_batch = {}
        
        for i, (input_contract, target_contract) in enumerate(selected_pairs):
            # Tokenize input and target
            input_encoding = self.augmenter.tokenizer(
                input_contract,
                max_length=batch['input_ids'].size(1),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            target_encoding = self.augmenter.tokenizer(
                target_contract,
                max_length=batch['input_ids'].size(1),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Create dummy AST data
            ast_input_ids = input_encoding['input_ids'].clone()
            ast_attention_mask = input_encoding['attention_mask'].clone()
            
            # PRESERVE the original vulnerability data instead of creating dummy zeros
            # Use the vulnerability data from the original batch
            if i < len(original_contract_vulns):
                contract_vulnerabilities = original_contract_vulns[i:i+1]  # Keep original vulnerability data
                vulnerable_lines = original_vulnerable_lines[i:i+1]  # Keep original vulnerability data
                token_to_line = original_token_to_line[i:i+1]  # Keep original token-to-line mapping
            else:
                # If we have more augmented samples than original, use the last original sample's data
                contract_vulnerabilities = original_contract_vulns[-1:]  # Use last original sample
                vulnerable_lines = original_vulnerable_lines[-1:]  # Use last original sample
                token_to_line = original_token_to_line[-1:]  # Use last original sample
            
            if i == 0:
                # Initialize new batch tensors
                new_batch['input_ids'] = input_encoding['input_ids']
                new_batch['attention_mask'] = input_encoding['attention_mask']
                new_batch['ast_input_ids'] = ast_input_ids
                new_batch['ast_attention_mask'] = ast_attention_mask
                new_batch['target_ids'] = target_encoding['input_ids']
                new_batch['vulnerable_lines'] = vulnerable_lines
                new_batch['contract_vulnerabilities'] = contract_vulnerabilities
                new_batch['token_to_line'] = token_to_line
                new_batch['source_code'] = [input_contract]
                new_batch['contract_name'] = [f"{contract_name}_augmented_{i}"]
            else:
                # Concatenate to existing tensors
                new_batch['input_ids'] = torch.cat([new_batch['input_ids'], input_encoding['input_ids']], dim=0)
                new_batch['attention_mask'] = torch.cat([new_batch['attention_mask'], input_encoding['attention_mask']], dim=0)
                new_batch['ast_input_ids'] = torch.cat([new_batch['ast_input_ids'], ast_input_ids], dim=0)
                new_batch['ast_attention_mask'] = torch.cat([new_batch['ast_attention_mask'], ast_attention_mask], dim=0)
                new_batch['target_ids'] = torch.cat([new_batch['target_ids'], target_encoding['input_ids']], dim=0)
                new_batch['vulnerable_lines'] = torch.cat([new_batch['vulnerable_lines'], vulnerable_lines], dim=0)
                new_batch['contract_vulnerabilities'] = torch.cat([new_batch['contract_vulnerabilities'], contract_vulnerabilities], dim=0)
                new_batch['token_to_line'] = torch.cat([new_batch['token_to_line'], token_to_line], dim=0)
                new_batch['source_code'].append(input_contract)
                new_batch['contract_name'].append(f"{contract_name}_augmented_{i}")
        
        return new_batch

    def train_epoch(self, epoch):
        self.model.train()
        
        total_gen_loss = 0
        total_contract_vuln_loss = 0
        total_line_vuln_loss = 0
        total_discriminator_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            desc=f'Epoch {epoch}',
            bar_format='{l_bar}{bar:10}{r_bar}'
        )
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Apply augmentation if enabled
                if self.use_augmentation:
                    batch = self._create_augmented_batch(batch)
                
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
                
                # Handle target_ids based on augmentation setting
                if self.use_augmentation:
                    target_ids = batch['target_ids']
                else:
                    target_ids = input_ids
                
                # Forward pass
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        target_ids=target_ids,
                        token_to_line=token_to_line
                    )
                    
                except Exception as e:
                    print(f"Error in model forward pass: {str(e)}")
                    continue
                
                # Get vulnerability predictions
                contract_vuln_logits = outputs['contract_vulnerability_logits']
                line_vuln_logits = outputs['line_vulnerability_logits']
                
                # Calculate generation loss
                try:
                    logits = outputs['logits']
                    target_ids_shifted = outputs['target_ids']
                    gen_loss = self.generator_loss_fn(logits, target_ids_shifted)
                except KeyError as e:
                    print(f"Missing key in outputs: {e}")
                    continue
                except Exception as e:
                    print(f"Error calculating generation loss: {str(e)}")
                    continue
                
                # Calculate contract-level vulnerability loss
                contract_vuln_loss = self.focal_loss(
                    contract_vuln_logits,
                    contract_vulnerabilities.float()
                )
                
                # IMPROVED: Calculate line-level vulnerability loss with spatial awareness
                line_vuln_loss = self.spatial_focal_loss(
                    line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                    vulnerable_lines.view(-1, self.model.num_vulnerability_types).float(),
                    token_to_line.view(-1) if token_to_line is not None else None
                )
                
                # Dynamic focal loss adjustment based on batch vulnerability distribution
                batch_contract_vulns = contract_vulnerabilities.sum().item()
                batch_line_vulns = vulnerable_lines.sum().item()
                batch_size = contract_vulnerabilities.size(0)
                
                # Adjust focal loss alpha based on vulnerability rate in this batch
                if batch_contract_vulns > 0:
                    # If we have vulnerabilities, use more aggressive focal loss
                    self.focal_loss.alpha = 0.05  # More focus on positive cases
                    self.focal_loss.gamma = 4.0  # More aggressive down-weighting
                else:
                    # If no vulnerabilities, use balanced focal loss
                    self.focal_loss.alpha = 0.25
                    self.focal_loss.gamma = 2.0
                
                # IMPROVED: Dynamic line vulnerability loss adjustment
                if batch_line_vulns > 0:
                    # If we have line vulnerabilities, use very aggressive focal loss
                    self.spatial_focal_loss.alpha = 0.005  # Very low alpha for extreme imbalance
                    self.spatial_focal_loss.gamma = 5.0  # Very aggressive down-weighting
                    self.spatial_focal_loss.spatial_weight = 0.5  # Higher spatial weight
                else:
                    # If no line vulnerabilities, use standard settings
                    self.spatial_focal_loss.alpha = 0.01
                    self.spatial_focal_loss.gamma = 4.0
                    self.spatial_focal_loss.spatial_weight = 0.3
                
                # Apply minimum loss thresholds to prevent losses from going to zero
                contract_vuln_loss = torch.max(contract_vuln_loss, torch.tensor(0.0001).to(self.device))
                line_vuln_loss = torch.max(line_vuln_loss, torch.tensor(0.00001).to(self.device))  # Increased from 0.01
                
                # GAN training with integrated discriminator
                discriminator_loss = 0.0
                adversarial_loss = 0.0
                discriminator_confidence = 0.5  # Default value
                
                if self.use_gan:
                    try:
                        # Get discriminator logits from model output
                        discriminator_logits = outputs['discriminator_logits']
                        
                        if discriminator_logits is not None:
                            # Create labels: 1 for real contracts, 0 for generated contracts
                            batch_size = discriminator_logits.size(0)
                            real_labels = torch.ones(batch_size, 1).to(self.device)  # 1 = real
                            fake_labels = torch.zeros(batch_size, 1).to(self.device)  # 0 = fake
                            
                            # Calculate discriminator loss (real vs fake classification)
                            # We want the discriminator to output high values for real contracts
                            discriminator_loss = self.adversarial_loss(discriminator_logits, real_labels)
                            
                            # Add small adversarial penalty to generator when discriminator is very confident
                            discriminator_confidence = torch.sigmoid(discriminator_logits).mean().item()
                            
                            # If discriminator easily spots fakes (confidence < 0.3), add small penalty to generator
                            if discriminator_confidence < 0.3:
                                # Small adversarial loss to make generator fool the discriminator
                                adversarial_loss = self.adversarial_loss(discriminator_logits, fake_labels)
                            
                            # Add gradient penalty to prevent discriminator from becoming too confident
                            if discriminator_confidence > 0.8:
                                # Penalize discriminator for being too confident (prevent mode collapse)
                                confidence_penalty = torch.mean(torch.sigmoid(discriminator_logits) ** 2)
                                discriminator_loss = discriminator_loss + 1.0 * confidence_penalty  # Increased from 0.5
                            
                            # Add stronger regularization to prevent mode collapse
                            if discriminator_confidence > 0.8:  # Reduced threshold from 0.95
                                # Very strong penalty for extreme overconfidence
                                extreme_penalty = torch.mean(torch.sigmoid(discriminator_logits) ** 4)
                                discriminator_loss = discriminator_loss + 2.0 * extreme_penalty  # Increased from 1.0
                        
                    except KeyError as e:
                        print(f"Missing discriminator_logits in model outputs: {e}")
                        print(f"Available keys: {list(outputs.keys())}")
                    except Exception as e:
                        print(f"Error in GAN training: {str(e)}")
                        continue
                
                # IMPROVED: Combined loss with much higher weights for line vulnerabilities
                if self.use_augmentation and self.use_gan:
                    # Much higher weights for vulnerability detection, especially line-level
                    total_loss = (
                        0.1 * gen_loss +  # Reduced from 0.15
                        0.2 * contract_vuln_loss * 20 +  # Reduced from 0.4
                        0.6 * line_vuln_loss * self.line_vuln_weight +  # Much higher weight for line vulnerabilities
                        0.1 * discriminator_loss  # Reduced from 0.15
                    )
                elif self.use_augmentation:
                    total_loss = (
                        0.15 * gen_loss +  # Reduced from 0.2
                        0.25 * contract_vuln_loss * 20 +  # Reduced from 0.4
                        0.6 * line_vuln_loss * self.line_vuln_weight  # Much higher weight for line vulnerabilities
                    )
                else:
                    total_loss = (
                        0.1 * gen_loss +  # Reduced from 0.15
                        0.2 * contract_vuln_loss * 20 +  # Reduced from 0.4
                        0.7 * line_vuln_loss * self.line_vuln_weight  # Much higher weight for line vulnerabilities
                    )
                
                # Add GAN losses to total loss
                if self.use_gan:
                    # Add adversarial loss if discriminator is too confident
                    if adversarial_loss > 0:
                        total_loss = total_loss + 0.05 * adversarial_loss
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Additional gradient clipping for different components
                if self.use_gan:
                    # Clip discriminator gradients separately to prevent them from overwhelming other components
                    discriminator_params = [p for name, p in self.model.named_parameters() if 'disc_' in name]
                    if discriminator_params:
                        torch.nn.utils.clip_grad_norm_(discriminator_params, self.max_grad_norm * 0.3)  # Reduced from 0.5
                
                # Clip vulnerability head gradients separately
                vuln_params = [p for name, p in self.model.named_parameters() 
                             if 'vulnerability_head' in name or 'spatial_attention' in name]
                if vuln_params:
                    torch.nn.utils.clip_grad_norm_(vuln_params, self.max_grad_norm * 1.5)  # Increased for line vulnerability heads
                
                # Check for gradient explosion after clipping
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN or Inf loss detected. Skipping update.")
                    self.optimizer.zero_grad()
                    continue

                if total_norm > 1000:
                    print(f"Extremely high gradient norm detected: {total_norm:.4f}. Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                # Update metrics
                total_gen_loss += gen_loss.item()
                total_contract_vuln_loss += contract_vuln_loss.item()
                total_line_vuln_loss += line_vuln_loss.item()
                total_discriminator_loss += discriminator_loss
                batch_count += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'contract_vuln_loss': f'{contract_vuln_loss.item():.4f}',
                    'line_vuln_loss': f'{line_vuln_loss.item():.6f}',  # More precision for line loss
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'grad_norm': f'{total_norm:.2f}',
                    'aug': 'ON' if self.use_augmentation else 'OFF',
                    'gan': 'ON' if self.use_gan else 'OFF',
                    'disc_loss': f'{discriminator_loss:.4f}' if self.use_gan else 'N/A',
                    'disc_conf': f'{discriminator_confidence:.3f}' if self.use_gan else 'N/A',
                    'line_weight': f'{self.line_vuln_weight:.0f}'  # Show line vulnerability weight
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
        
        progress_bar.close()
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'contract_vuln_loss': total_contract_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'line_vuln_loss': total_line_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'discriminator_loss': total_discriminator_loss / batch_count if batch_count > 0 else 0.0
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
                    # Apply augmentation if enabled (but with fewer variants for validation)
                    if self.use_augmentation:
                        batch = self._create_augmented_batch(batch)
                    
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
                    
                    # Handle target_ids based on augmentation setting
                    if self.use_augmentation:
                        target_ids = batch['target_ids']
                    else:
                        target_ids = input_ids
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        target_ids=target_ids,
                        token_to_line=token_to_line
                    )
                    
                    # Get vulnerability predictions
                    contract_vuln_logits = outputs['contract_vulnerability_logits']
                    line_vuln_logits = outputs['line_vulnerability_logits']
                    
                    # Calculate generation loss (language modeling loss)
                    try:
                        logits = outputs['logits']
                        target_ids_shifted = outputs['target_ids']
                        gen_loss = self.generator_loss_fn(logits, target_ids_shifted)
                    except KeyError as e:
                        print(f"Missing key in outputs: {e}")
                        print(f"Available keys: {list(outputs.keys())}")
                        continue
                    except Exception as e:
                        print(f"Error calculating generation loss: {str(e)}")
                        continue
                    
                    # Calculate contract-level vulnerability loss
                    contract_vuln_loss = self.focal_loss(
                        contract_vuln_logits,
                        contract_vulnerabilities.float()
                    )
                    
                    # IMPROVED: Calculate line-level vulnerability loss with spatial awareness
                    line_vuln_loss = self.spatial_focal_loss(
                        line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                        vulnerable_lines.view(-1, self.model.num_vulnerability_types).float(),
                        token_to_line.view(-1) if token_to_line is not None else None
                    )
                    
                    # Apply minimum loss thresholds to prevent losses from going to zero
                    contract_vuln_loss = torch.max(contract_vuln_loss, torch.tensor(0.0001).to(self.device))
                    line_vuln_loss = torch.max(line_vuln_loss, torch.tensor(0.00001).to(self.device))  # Increased from 0.01
                    
                    # IMPROVED: Combined loss with higher weights for line vulnerabilities
                    if self.use_augmentation and self.use_gan:
                        total_loss = (
                            0.4 * gen_loss +
                            0.25 * contract_vuln_loss +
                            0.35 * line_vuln_loss * self.line_vuln_weight  # Use dynamic weight
                        )
                    elif self.use_augmentation:
                        total_loss = (
                            0.6 * gen_loss +
                            0.25 * contract_vuln_loss +
                            0.15 * line_vuln_loss * self.line_vuln_weight  # Use dynamic weight
                        )
                    else:
                        total_loss = (
                            0.4 *gen_loss + 
                            0.3 * contract_vuln_loss +
                            0.3 * line_vuln_loss * self.line_vuln_weight  # Use dynamic weight
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
            'line_vuln_loss': total_line_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'discriminator_loss': 0.0
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
            
            if self.use_gan:
                self.training_history['discriminator_loss'].append(train_metrics['discriminator_loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Contract Vulnerability Loss: {train_metrics['contract_vuln_loss']:.4f}")
            print(f"Line Vulnerability Loss: {train_metrics['line_vuln_loss']:.4f}")
            if self.use_gan:
                print(f"Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Step the scheduler with the validation loss
            self.scheduler.step(val_metrics['gen_loss'])
            
            # Check if learning rate is too small and boost it
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-6 and self.patience_counter > 5:
                print(f"⚠️  Learning rate too small ({current_lr:.8f}). Boosting to 1e-5...")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-5
                # Reset patience to give it another chance
                self.patience_counter = 0
            
            # Additional learning rate boost if still stuck
            if current_lr < 1e-6 and self.patience_counter > 10:
                print(f"🚨 Learning rate critically low ({current_lr:.8f}). Boosting to 5e-5...")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 5e-5
                # Reset patience to give it another chance
                self.patience_counter = 0
            
            # Check for validation loss explosion
            if val_metrics['gen_loss'] > self.best_val_loss * 2.0:  # If validation loss doubles
                print(f"⚠️  Validation loss explosion detected! Current: {val_metrics['gen_loss']:.4f}, Best: {self.best_val_loss:.4f}")
                print("Reducing learning rate by 10x to stabilize training...")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                # Reset patience to give it another chance
                self.patience_counter = 0
            
            # Save checkpoint if validation loss improved
            if val_metrics['gen_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['gen_loss']
                self.patience_counter = 0
                
                # Add augmentation and GAN suffixes to checkpoint name
                suffix = ""
                if self.use_augmentation:
                    suffix += "_augmented"
                if self.use_gan:
                    suffix += "_gan"
                
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model{suffix}_epoch_{epoch + 1}.pt')
                
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['gen_loss'],
                    'training_history': self.training_history,
                    'use_augmentation': self.use_augmentation,
                    'use_gan': self.use_gan
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"🎉 New best validation loss! Saved checkpoint to {checkpoint_path}")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                break
            
            # Save latest checkpoint
            suffix = ""
            if self.use_augmentation:
                suffix += "_augmented"
            if self.use_gan:
                suffix += "_gan"
            
            latest_checkpoint_path = os.path.join(checkpoint_dir, f'latest_model{suffix}.pt')
            
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_metrics['gen_loss'],
                'training_history': self.training_history,
                'use_augmentation': self.use_augmentation,
                'use_gan': self.use_gan
            }
            
            torch.save(checkpoint_data, latest_checkpoint_path)