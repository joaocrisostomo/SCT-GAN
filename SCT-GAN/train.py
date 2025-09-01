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
import re

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
    def __init__(self, alpha=0.25, gamma=2.0, spatial_weight=0.2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spatial_weight = spatial_weight
        self.reduction = reduction
        
    def forward(self, pred, target, token_to_line=None):
        # NEW: Custom loss that encourages positive predictions for vulnerable lines
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        # Calculate standard focal loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # NEW: Add positive prediction encouragement (less aggressive)
        # For vulnerable lines (target == 1), encourage higher probabilities
        vulnerable_mask = (target == 1.0)
        if vulnerable_mask.any():
            # Encourage vulnerable lines to have higher probabilities (less aggressive)
            prob_encouragement = torch.where(
                vulnerable_mask,
                torch.relu(0.3 - probs) * 0.5,  # Reduced penalty and threshold
                torch.tensor(0.0, device=pred.device)
            )
            focal_loss = focal_loss + prob_encouragement
        
        # NEW: Add negative prediction discouragement (less aggressive)
        # For non-vulnerable lines (target == 0), discourage very high probabilities
        non_vulnerable_mask = (target == 0.0)
        if non_vulnerable_mask.any():
            # Discourage non-vulnerable lines from having very high probabilities (less aggressive)
            prob_discouragement = torch.where(
                non_vulnerable_mask,
                torch.relu(probs - 0.5) * 0.2,  # Reduced penalty and increased threshold
                torch.tensor(0.0, device=pred.device)
            )
            focal_loss = focal_loss + prob_discouragement
        
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
        # Handle the case where token_to_line is None
        if token_to_line is None:
            return torch.zeros_like(pred)
        
        # Reshape back to batch format for spatial computation
        # The input is flattened, so we need to determine the batch size
        total_tokens = pred.shape[0]
        
        # Try to determine batch size from token_to_line shape
        if token_to_line.shape[0] == total_tokens:
            # If token_to_line has the same number of tokens, assume batch_size=1
            batch_size = 1
            seq_len = total_tokens
        else:
            # Otherwise, try to infer from total tokens
            if total_tokens % 1024 == 0:
                batch_size = total_tokens // 1024
                seq_len = 1024
            else:
                # Fallback: assume single batch
                batch_size = 1
                seq_len = total_tokens
        
        num_classes = pred.shape[1]
        device = pred.device
        
        # Ensure we have valid shapes
        if batch_size * seq_len != total_tokens:
            # If shapes don't match, return zero penalty
            return torch.zeros_like(pred)
        
        # Reshape tensors to [batch_size, seq_len, num_classes]
        try:
            pred_reshaped = pred.view(batch_size, seq_len, num_classes)
            target_reshaped = target.view(batch_size, seq_len, num_classes)
            token_to_line_reshaped = token_to_line.view(batch_size, seq_len)
        except Exception:
            # If reshaping fails, return zero penalty
            return torch.zeros_like(pred)
        
        # Create spatial penalty tensor
        spatial_penalty = torch.zeros_like(pred_reshaped)
        
        for b in range(batch_size):
            for i in range(seq_len):
                try:
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
                except Exception:
                    # If there's an error processing this token, skip it
                    continue
        
        # Flatten back to match input shape
        return spatial_penalty.view(-1, num_classes)

class SoliditySyntaxLoss(nn.Module):
    """
    Loss function that penalizes the generator for generating invalid Solidity syntax.
    This helps ensure generated contracts are syntactically correct.
    """
    def __init__(self, tokenizer, syntax_weight=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.syntax_weight = syntax_weight
        
        # Pre-compute token IDs for common Solidity tokens
        self._init_token_mappings()
        
    def _init_token_mappings(self):
        """Initialize mappings for common Solidity tokens"""
        # Common Solidity keywords that should be followed by specific tokens
        self.keyword_followers = {
            'function': ['(', 'view', 'pure', 'external', 'public', 'internal', 'private'],
            'contract': ['{', 'is', 'interface'],
            'if': ['('],
            'for': ['('],
            'while': ['('],
            'require': ['('],
            'assert': ['('],
            'revert': ['('],
            'emit': ['('],
            'return': [';', '('],
            'break': [';'],
            'continue': [';'],
            'import': ['"', "'"],
            'pragma': ['solidity'],
            'struct': ['{'],
            'enum': ['{'],
            'event': ['('],
            'modifier': ['{', '('],
            'mapping': ['('],
        }
        
        # Convert keywords to token IDs
        self.keyword_token_ids = {}
        for keyword in self.keyword_followers.keys():
            token_id = self.tokenizer.convert_tokens_to_ids(keyword)
            if token_id != self.tokenizer.unk_token_id:
                self.keyword_token_ids[token_id] = keyword
        
        # Convert follower tokens to token IDs
        self.follower_token_ids = {}
        for keyword, followers in self.keyword_followers.items():
            keyword_id = self.tokenizer.convert_tokens_to_ids(keyword)
            if keyword_id != self.tokenizer.unk_token_id:
                follower_ids = []
                for follower in followers:
                    follower_id = self.tokenizer.convert_tokens_to_ids(follower)
                    if follower_id != self.tokenizer.unk_token_id:
                        follower_ids.append(follower_id)
                if follower_ids:
                    self.follower_token_ids[keyword_id] = follower_ids
        
        # Common token IDs
        self.semicolon_id = self.tokenizer.convert_tokens_to_ids(';')
        self.open_paren_id = self.tokenizer.convert_tokens_to_ids('(')
        self.close_paren_id = self.tokenizer.convert_tokens_to_ids(')')
        self.open_brace_id = self.tokenizer.convert_tokens_to_ids('{')
        self.close_brace_id = self.tokenizer.convert_tokens_to_ids('}')
        
        print(f"✓ Syntax loss initialized with {len(self.keyword_token_ids)} keywords")
        
    def forward(self, logits, target_ids, generated_sequence=None):
        """
        Calculate syntax-aware loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size] or [batch_size*seq_len, vocab_size]
            target_ids: Target token IDs [batch_size, seq_len] or [batch_size*seq_len]
            generated_sequence: Generated sequence for syntax analysis (optional)
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='mean')
        
        # Syntax penalty - much simpler and more effective
        syntax_penalty = self._compute_simple_syntax_penalty(logits, target_ids)
        
        # Combine losses
        total_loss = ce_loss + self.syntax_weight * syntax_penalty
        
        return total_loss
    
    def _compute_simple_syntax_penalty(self, logits, target_ids):
        """Compute simple but effective syntax penalty"""
        try:
            # Handle both 2D and 3D logits
            if logits.dim() == 2:
                # Flattened: [batch_size * seq_len, vocab_size]
                total_tokens = target_ids.size(0)
                if total_tokens % 1024 == 0:
                    batch_size = total_tokens // 1024
                    seq_len = 1024
                else:
                    batch_size = 1
                    seq_len = total_tokens
                
                try:
                    logits_3d = logits.view(batch_size, seq_len, -1)
                    target_3d = target_ids.view(batch_size, seq_len)
                except:
                    return torch.tensor(0.0, device=logits.device)
            else:
                # Already 3D: [batch_size, seq_len, vocab_size]
                logits_3d = logits
                target_3d = target_ids
                batch_size, seq_len = target_3d.shape[:2]
            
            device = logits.device
            total_penalty = 0.0
            penalty_count = 0
            
            # Get token IDs with null checks
            return_id = self.tokenizer.convert_tokens_to_ids('return')
            break_id = self.tokenizer.convert_tokens_to_ids('break')
            continue_id = self.tokenizer.convert_tokens_to_ids('continue')
            
            # Only proceed if we have valid token IDs
            if return_id is None or break_id is None or continue_id is None:
                return torch.tensor(0.0, device=device)
            
            # Check if other token IDs are valid
            if (self.semicolon_id is None or self.open_paren_id is None or 
                self.close_paren_id is None or self.open_brace_id is None or 
                self.close_brace_id is None):
                return torch.tensor(0.0, device=device)
            
            for b in range(batch_size):
                for i in range(seq_len - 1):  # Look at current and next token
                    current_token = target_3d[b, i].item()
                    next_token = target_3d[b, i + 1].item()
                    
                    # Check for keyword-follower violations
                    if current_token in self.keyword_token_ids:
                        keyword = self.keyword_token_ids[current_token]
                        expected_followers = self.follower_token_ids.get(current_token, [])
                        
                        if expected_followers and next_token not in expected_followers:
                            # Apply penalty for incorrect follower
                            total_penalty += 2.0
                            penalty_count += 1
                    
                    # Check for missing semicolons after statements
                    if current_token in [return_id, break_id, continue_id]:
                        if next_token != self.semicolon_id:
                            total_penalty += 1.5
                            penalty_count += 1
                    
                    # Check for balanced parentheses
                    if current_token == self.open_paren_id:
                        # Look ahead for matching close parenthesis
                        found_match = False
                        for j in range(i + 1, min(i + 20, seq_len)):
                            if target_3d[b, j].item() == self.close_paren_id:
                                found_match = True
                                break
                        if not found_match:
                            total_penalty += 1.0
                            penalty_count += 1
                    
                    # Check for balanced braces
                    if current_token == self.open_brace_id:
                        # Look ahead for matching close brace
                        found_match = False
                        for j in range(i + 1, min(i + 50, seq_len)):
                            if target_3d[b, j].item() == self.close_brace_id:
                                found_match = True
                                break
                        if not found_match:
                            total_penalty += 1.0
                            penalty_count += 1
            
            # Return average penalty
            if penalty_count > 0:
                return torch.tensor(total_penalty / penalty_count, device=device)
            else:
                return torch.tensor(0.0, device=device)
                
        except Exception as e:
            print(f"Error in syntax penalty calculation: {str(e)}")
            return torch.tensor(0.0, device=logits.device)

class ContractLevelFocalLoss(nn.Module):
    """
    Specialized focal loss for contract-level vulnerability detection.
    Handles extreme class imbalance and provides better learning for rare vulnerability types.
    """
    def __init__(self, alpha=0.1, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        # Calculate focal loss for each vulnerability type
        focal_losses = []
        for i in range(pred.size(1)):  # For each vulnerability type
            pred_i = pred[:, i]
            target_i = target[:, i]
            
            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(pred_i, target_i, reduction='none')
            
            # Focal loss calculation
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            
            # Add extra penalty for missed vulnerabilities (false negatives)
            false_negative_penalty = torch.where(
                (target_i == 1) & (probs[:, i] < 0.5),
                torch.tensor(2.0).to(pred.device),  # Higher penalty for missed vulnerabilities
                torch.tensor(1.0).to(pred.device)
            )
            
            focal_loss = focal_loss * false_negative_penalty
            focal_losses.append(focal_loss)
        
        # Stack all vulnerability types
        total_loss = torch.stack(focal_losses, dim=1)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss

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

        # Initialize syntax-aware loss for training
        self.syntax_loss = SoliditySyntaxLoss(tokenizer, syntax_weight=0.5)
        
        # Get base model parameters (excluding vulnerability heads)
        base_params = []
        contract_head_params = []
        line_head_params = []
        discriminator_params = []
        
        for name, param in self.model.named_parameters():
            if 'disc_' in name and self.use_gan:
                discriminator_params.append(param)
            elif 'contract_vulnerability_head' in name or 'contract_feature_aggregation' in name or 'contract_vuln_attention' in name:
                contract_head_params.append(param)
            elif ('line_vulnerability_head' in name or 'line_feature_extractor' in name or 
                  'line_vuln_attention' in name or 'vuln_type_attention' in name):
                line_head_params.append(param)
            else:
                base_params.append(param)
        
        # IMPROVED: Initialize optimizer with balanced learning rates
        param_groups = [
            {'params': base_params, 'lr': learning_rate},
            {'params': contract_head_params, 'lr': learning_rate * 2.0},  # Moderate LR for contract vulnerability heads
            {'params': line_head_params, 'lr': learning_rate * 3.0},  # Conservative LR for line vulnerability heads
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
        
        # NEW: Specialized contract-level focal loss
        self.contract_focal_loss = ContractLevelFocalLoss(
            alpha=0.05,  # Very low alpha for extreme class imbalance
            gamma=4.0,   # Higher gamma for more aggressive down-weighting
            reduction='mean'
        )
        
        # IMPROVED: Spatial-aware focal loss for line vulnerabilities with balanced parameters
        self.spatial_focal_loss = SpatialAwareFocalLoss(
            alpha=0.25,  # Standard focal loss alpha
            gamma=2.0,   # Standard focal loss gamma
            spatial_weight=0.2,  # Moderate spatial weight
            reduction='mean'
        )
        
        # IMPROVED: Generator loss with syntax awareness
        self.generator_loss_fn = self.syntax_loss  # Use syntax-aware loss instead of simple CrossEntropyLoss
        
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
            'discriminator_loss': [],
            'syntax_loss': [],  # NEW: Track syntax loss
            'line_vuln_accuracy': [],  # NEW: Track line vulnerability accuracy
            'line_vuln_precision': [],  # NEW: Track line vulnerability precision
            'line_vuln_recall': []  # NEW: Track line vulnerability recall
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
        
        # NEW: Analyze vulnerability type distribution
        vulnerability_type_counts = [0] * 8  # 8 vulnerability types
        
        for batch in self.train_dataloader:
            batch_size = batch['contract_vulnerabilities'].size(0)
            total_contracts += batch_size
            
            # Count contract-level vulnerabilities
            contract_vulns = batch['contract_vulnerabilities'].sum().item()
            total_contract_vulns += contract_vulns
            
            # Count line-level vulnerabilities
            line_vulns = batch['vulnerable_lines'].sum().item()
            total_line_vulns += line_vulns
            
            # Count vulnerability types
            for i in range(8):
                vulnerability_type_counts[i] += batch['contract_vulnerabilities'][:, i].sum().item()
        
        contract_vuln_rate = total_contract_vulns / total_contracts if total_contracts > 0 else 0
        line_vuln_rate = total_line_vulns / (total_contracts * 1024 * 8) if total_contracts > 0 else 0  # 1024 seq_len * 8 vuln_types
        
        print(f"Total contracts: {total_contracts}")
        print(f"Total contract vulnerabilities: {total_contract_vulns}")
        print(f"Total line vulnerabilities: {total_line_vulns}")
        print(f"Contract vulnerability rate: {contract_vuln_rate:.4f} ({contract_vuln_rate*100:.2f}%)")
        print(f"Line vulnerability rate: {line_vuln_rate:.6f} ({line_vuln_rate*100:.4f}%)")
        
        # NEW: Print vulnerability type distribution
        print("\n=== Vulnerability Type Distribution ===")
        vulnerability_types = ['ARTHM', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'Tx-Origin', 'UE']
        for i, (vuln_type, count) in enumerate(zip(vulnerability_types, vulnerability_type_counts)):
            rate = count / total_contracts if total_contracts > 0 else 0
            print(f"{vuln_type}: {count} ({rate*100:.2f}%)")
        
        # Store vulnerability rates for dynamic weighting
        self.contract_vuln_rate = contract_vuln_rate
        self.line_vuln_rate = line_vuln_rate
        self.vulnerability_type_counts = vulnerability_type_counts
        
        # NEW: Warm-up tracking for line vulnerability detection
        self.current_epoch = 0
        self.warmup_epochs = 5  # Warm up over 5 epochs
        
        # NEW: Stability tracking to prevent oscillation
        self.prev_line_recall = 0.0
        self.prev_line_precision = 0.0
        self.stability_factor = 1.0
        
        # NEW: Add oscillation detection for line vulnerability logits
        self.prev_line_logit_mean = 0.0
        self.prev_line_logit_std = 0.0
        self.oscillation_detected = False
        self.consecutive_oscillations = 0
        
        # NEW: Add adaptive loss scaling
        self.line_loss_scale = 1.0
        self.min_line_loss_scale = 0.1
        self.max_line_loss_scale = 5.0
        self.loss_warmup_epochs = 5  # Number of epochs to keep line_loss_scale at 1.0
        
        # NEW: Add prediction tracking
        self.total_line_predictions = 0
        self.batches_with_predictions = 0
        
        # IMPROVED: Much more balanced dynamic loss weights based on vulnerability rates
        if line_vuln_rate < 0.001:  # Extreme imbalance
            self.line_vuln_weight = 5.0  # Much more conservative weight
            print(f"⚠️  Extreme line vulnerability imbalance detected. Using weight: {self.line_vuln_weight}")
        elif line_vuln_rate < 0.01:  # High imbalance
            self.line_vuln_weight = 3.0  # More conservative weight
            print(f"⚠️  High line vulnerability imbalance detected. Using weight: {self.line_vuln_weight}")
        else:
            self.line_vuln_weight = 2.0  # Conservative weight
            print(f"Line vulnerability weight: {self.line_vuln_weight}")
        
        self.contract_vuln_weight = 3.0  # Conservative contract weight

        print("✓ Syntax-aware loss enabled - will penalize invalid Solidity syntax")
        
        # NEW: Test model dimensions to catch issues early
        print("\n=== Model Dimension Test ===")
        try:
            # Create dummy inputs
            test_input_ids = torch.randint(0, 1000, (2, 1024)).to(self.device)
            test_attention_mask = torch.ones(2, 1024).to(self.device)
            test_ast_input_ids = torch.randint(0, 1000, (2, 1024)).to(self.device)
            test_ast_attention_mask = torch.ones(2, 1024).to(self.device)
            test_token_to_line = torch.randint(0, 100, (2, 1024)).to(self.device)
            
            # NEW: Enable debug mode temporarily
            self.model._debug_mode = True
            
            with torch.no_grad():
                test_outputs = self.model(
                    input_ids=test_input_ids,
                    attention_mask=test_attention_mask,
                    ast_input_ids=test_ast_input_ids,
                    ast_attention_mask=test_ast_attention_mask,
                    target_ids=test_input_ids,
                    token_to_line=test_token_to_line
                )
                
                # Disable debug mode after test
                self.model._debug_mode = False
                
                print(f"✓ Model forward pass successful")
                print(f"✓ Contract vuln logits: {test_outputs['contract_vulnerability_logits'].shape}")
                print(f"✓ Line vuln logits: {test_outputs['line_vulnerability_logits'].shape}")
                print(f"✓ Expected contract shape: [2, 8]")
                print(f"✓ Expected line shape: [2, 1024, 8]")
                
                # NEW: Test line vulnerability head outputs
                line_logits = test_outputs['line_vulnerability_logits']
                print(f"✓ Line logits range: [{line_logits.min().item():.4f}, {line_logits.max().item():.4f}]")
                print(f"✓ Line logits mean: {line_logits.mean().item():.4f}")
                print(f"✓ Line logits std: {line_logits.std().item():.4f}")
                
                # Check if line logits are all the same (indicating dead neurons)
                if line_logits.std().item() < 1e-6:
                    print("⚠️  WARNING: Line vulnerability logits have very low variance!")
                    print("This suggests the line vulnerability heads might not be properly initialized")
                else:
                    print("✓ Line vulnerability heads appear to be working correctly")
                
                # NEW: Test if line vulnerability heads can produce meaningful outputs
                print(f"✓ Line logits after sigmoid: [{torch.sigmoid(line_logits).min().item():.6f}, {torch.sigmoid(line_logits).max().item():.6f}]")
                print(f"✓ Line logits predictions: {(torch.sigmoid(line_logits) > 0.5).sum().item()} out of {line_logits.numel()}")
                
                # Test with different thresholds
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    pred_count = (torch.sigmoid(line_logits) > threshold).sum().item()
                    print(f"✓ Threshold {threshold}: {pred_count} predictions")
                
                # NEW: Test if the model can produce varied outputs
                if line_logits.std().item() < 1e-3:
                    print("⚠️  WARNING: Line vulnerability logits still have very low variance!")
                    print("This suggests the model needs better initialization or architecture")
                else:
                    print("✓ Line vulnerability heads are producing varied outputs")
                
        except Exception as e:
            print(f"✗ Model dimension test failed: {str(e)}")
            raise e

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
        
        # Set current epoch in model for debugging
        self.model.set_current_epoch(epoch)
        
        total_gen_loss = 0
        total_contract_vuln_loss = 0
        total_line_vuln_loss = 0
        total_discriminator_loss = 0
        total_syntax_loss = 0
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
                
                # NEW: Calculate warm-up factor early in the training loop
                warmup_factor = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
                line_vuln_weight_adjusted = self.line_vuln_weight * warmup_factor * self.stability_factor * self.line_loss_scale
                
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
                    print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                    print(f"Input shapes - ast_input_ids: {ast_input_ids.shape}, ast_attention_mask: {ast_attention_mask.shape}")
                    continue
                
                # Get vulnerability predictions
                contract_vuln_logits = outputs['contract_vulnerability_logits']
                line_vuln_logits = outputs['line_vulnerability_logits']
                
                # Calculate generation loss
                try:
                    logits = outputs['logits']
                    target_ids_shifted = outputs['target_ids']
                    
                    # Note: Token constraints are disabled during training to avoid dimension issues
                    # The SoliditySyntaxLoss provides sufficient syntax guidance during training
                    # Token constraints can be applied during inference for better generation quality
                    
                    # Use syntax-aware loss instead of simple cross-entropy
                    # Pass None as the third argument since it's optional
                    gen_loss = self.generator_loss_fn(logits, target_ids_shifted, None)
                    
                    # Extract syntax penalty for tracking
                    syntax_penalty = None
                    if hasattr(self.generator_loss_fn, '_compute_simple_syntax_penalty'):
                        try:
                            syntax_penalty = self.generator_loss_fn._compute_simple_syntax_penalty(logits, target_ids_shifted)
                            total_syntax_loss += syntax_penalty.item()
                        except Exception as e:
                            print(f"Syntax penalty calculation failed: {str(e)}")
                            syntax_penalty = torch.tensor(0.0, device=logits.device)
                    
                except KeyError as e:
                    print(f"Missing key in outputs: {e}")
                    continue
                except Exception as e:
                    print(f"Error calculating generation loss: {str(e)}")
                    # Fallback to simple cross-entropy loss if syntax loss fails
                    try:
                        gen_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids_shifted.view(-1), reduction='mean')
                        syntax_penalty = torch.tensor(0.0, device=logits.device)
                        print(f"Fallback to simple cross-entropy loss successful")
                    except Exception as e2:
                        print(f"Fallback also failed: {str(e2)}")
                        continue
                
                # Calculate contract-level vulnerability loss
                contract_vuln_loss = self.contract_focal_loss(
                    contract_vuln_logits,
                    contract_vulnerabilities.float()
                )
                
                # IMPROVED: Calculate line-level vulnerability loss with spatial awareness
                # FIXED: Handle transposed dimensions for vulnerable_lines
                if line_vuln_logits.shape != vulnerable_lines.shape:
                    # Check if dimensions are transposed and fix them
                    if (line_vuln_logits.shape[0] == vulnerable_lines.shape[0] and 
                        line_vuln_logits.shape[1] == vulnerable_lines.shape[2] and 
                        line_vuln_logits.shape[2] == vulnerable_lines.shape[1]):
                        # Transpose vulnerable_lines to match line_vuln_logits
                        vulnerable_lines_for_loss = vulnerable_lines.transpose(1, 2).contiguous()
                    else:
                        vulnerable_lines_for_loss = vulnerable_lines
                else:
                    vulnerable_lines_for_loss = vulnerable_lines
                
                line_vuln_loss = self.spatial_focal_loss(
                    line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                    vulnerable_lines_for_loss.view(-1, self.model.num_vulnerability_types).float(),
                    token_to_line.view(-1) if token_to_line is not None else None
                )
                
                # NEW: Calculate line-level metrics for monitoring
                line_vuln_accuracy = 0.0
                line_vuln_precision = 0.0
                line_vuln_recall = 0.0
                
                # NEW: Debug line vulnerability logits
                if batch_idx == 0:  # Only print for first batch to avoid spam
                    print(f"\n=== Line Vulnerability Debug (Batch {batch_idx}) ===")
                    print(f"Line vuln logits shape: {line_vuln_logits.shape}")
                    print(f"Line vuln logits range: [{line_vuln_logits.min().item():.6f}, {line_vuln_logits.max().item():.6f}]")
                    print(f"Line vuln logits mean: {line_vuln_logits.mean().item():.6f}")
                    print(f"Line vuln logits std: {line_vuln_logits.std().item():.6f}")
                    print(f"Vulnerable lines shape: {vulnerable_lines.shape}")
                    print(f"Vulnerable lines sum: {vulnerable_lines.sum().item()}")
                    print(f"Vulnerable lines range: [{vulnerable_lines.min().item():.1f}, {vulnerable_lines.max().item():.1f}]")
                    
                    # NEW: Check for oscillation in line vulnerability logits
                    current_logit_mean = line_vuln_logits.mean().item()
                    current_logit_std = line_vuln_logits.std().item()
                    
                    # Only allow oscillation-based scaling after warmup epochs
                    if epoch >= self.loss_warmup_epochs:
                        if epoch > 0:
                            mean_change = abs(current_logit_mean - self.prev_line_logit_mean)
                            std_change = abs(current_logit_std - self.prev_line_logit_std)
                            if (mean_change > 5.0 or std_change > 1.0) and not self.oscillation_detected:  # Increased thresholds
                                print(f"⚠️  OSCILLATION DETECTED!")
                                print(f"Mean change: {mean_change:.3f} ({self.prev_line_logit_mean:.3f} → {current_logit_mean:.3f})")
                                print(f"Std change: {std_change:.3f} ({self.prev_line_logit_std:.3f} → {current_logit_std:.3f})")
                                self.oscillation_detected = True
                                self.consecutive_oscillations += 1
                                self.line_loss_scale = max(self.min_line_loss_scale, self.line_loss_scale * 0.5)  # Less aggressive reduction
                                print(f"Reduced line loss scale to: {self.line_loss_scale:.3f}")
                                self.stability_factor = max(0.5, self.stability_factor * 0.7)  # Less aggressive reduction
                                print(f"Reduced stability factor to: {self.stability_factor:.3f}")
                            self.prev_line_logit_mean = current_logit_mean
                            self.prev_line_logit_std = current_logit_std
                        else:
                            self.prev_line_logit_mean = current_logit_mean
                            self.prev_line_logit_std = current_logit_std
                    else:
                        # During warmup, keep line_loss_scale at 1.0
                        self.line_loss_scale = 1.0
                
                try:
                    # Convert logits to probabilities
                    line_vuln_probs = torch.sigmoid(line_vuln_logits)
                    
                    # NEW: Use adaptive thresholding instead of fixed 0.5
                    # Calculate threshold based on the top 1% of predictions (more conservative)
                    if line_vuln_probs.numel() > 0:
                        if line_vuln_logits.mean().item() < -1.0:
                            # If logits are negative, use a very low threshold
                            base_threshold = torch.quantile(line_vuln_probs, 0.99).item()
                            threshold = min(base_threshold, 0.4)  # Lower threshold
                            threshold = max(threshold, 0.1)  # Higher minimum
                            print(f"⚠️  Using conservative thresholds due to negative logits")
                        else:
                            base_threshold = torch.quantile(line_vuln_probs, 0.99).item()
                            threshold = min(base_threshold, 0.6)  # Higher threshold
                            threshold = max(threshold, 0.3)  # Higher minimum
                    
                    # Calculate binary predictions with adaptive threshold
                    line_vuln_preds = (line_vuln_probs > threshold).float()
                    
                    # NEW: Track predictions for monitoring
                    self.total_line_predictions += line_vuln_preds.sum().item()
                    if line_vuln_preds.sum().item() > 0:
                        self.batches_with_predictions += 1
                    
                    # NEW: Fallback mechanism - if too many predictions, use a higher threshold
                    if line_vuln_preds.sum().item() > 10000:  # If too many predictions
                        # Use a much higher threshold to be more selective
                        conservative_threshold = min(0.8, torch.quantile(line_vuln_probs, 0.995).item())
                        line_vuln_preds = (line_vuln_probs > conservative_threshold).float()
                        if batch_idx == 0:
                            print(f"⚠️  Too many predictions with threshold {threshold:.6f}, using conservative threshold {conservative_threshold:.6f}")
                            print(f"Conservative predictions: {line_vuln_preds.sum().item()}")
                    
                    # NEW: Ultra-fallback mechanism for very aggressive models
                    if line_vuln_preds.sum().item() > 5000:
                        # If still too many predictions, use very high threshold
                        ultra_conservative_threshold = min(0.9, torch.quantile(line_vuln_probs, 0.999).item())
                        line_vuln_preds = (line_vuln_probs > ultra_conservative_threshold).float()
                        if batch_idx == 0:
                            print(f"🚨  Still too many predictions! Using ultra-conservative threshold {ultra_conservative_threshold:.6f}")
                            print(f"Ultra-conservative predictions: {line_vuln_preds.sum().item()}")
                    
                    # NEW: Fallback mechanism - if no predictions, use a lower threshold
                    if line_vuln_preds.sum().item() == 0 and line_vuln_probs.max().item() > 0.1:
                        # If no predictions but we have any reasonable probabilities, use a much lower threshold
                        fallback_threshold = min(0.3, line_vuln_probs.max().item() * 0.5)
                        line_vuln_preds = (line_vuln_probs > fallback_threshold).float()
                        if batch_idx == 0:
                            print(f"⚠️  No predictions with threshold {threshold:.6f}, using fallback threshold {fallback_threshold:.6f}")
                            print(f"Fallback predictions: {line_vuln_preds.sum().item()}")
                    
                    # NEW: Ultra-fallback mechanism for very conservative models
                    if line_vuln_preds.sum().item() == 0:
                        # If still no predictions, use very low threshold
                        ultra_fallback_threshold = max(0.01, line_vuln_probs.max().item() * 0.3)
                        line_vuln_preds = (line_vuln_probs > ultra_fallback_threshold).float()
                        if batch_idx == 0:
                            print(f"🚨  Ultra-conservative model detected! Using ultra-fallback threshold {ultra_fallback_threshold:.6f}")
                            print(f"Ultra-fallback predictions: {line_vuln_preds.sum().item()}")
                            print(f"This will force some predictions to encourage learning")
                    
                    # NEW: Debug predictions
                    if batch_idx == 0:
                        print(f"Line vuln probs range: [{line_vuln_probs.min().item():.6f}, {line_vuln_probs.max().item():.6f}]")
                        print(f"Adaptive threshold: {threshold:.6f}")
                        print(f"Line vuln preds sum: {line_vuln_preds.sum().item()}")
                        print(f"Line vuln preds shape: {line_vuln_preds.shape}")
                        print(f"Top 1% probability: {torch.quantile(line_vuln_probs, 0.99).item():.6f}")
                        print(f"Top 0.5% probability: {torch.quantile(line_vuln_probs, 0.995).item():.6f}")
                        print(f"Top 0.1% probability: {torch.quantile(line_vuln_probs, 0.999).item():.6f}")
                        print(f"Warm-up factor: {warmup_factor:.3f}, Adjusted weight: {line_vuln_weight_adjusted:.1f}")
                        print(f"Probabilities above threshold: {(line_vuln_probs > threshold).sum().item()}")
                        print(f"Threshold selection: min={max(0.1, torch.quantile(line_vuln_probs, 0.99).item()):.6f}, max=0.6, final={threshold:.6f}")
                        print(f"Stability factor: {self.stability_factor:.3f}")
                    
                    # FIXED: Handle the shape mismatch - targets are transposed
                    if line_vuln_preds.shape != vulnerable_lines.shape:
                        # Check if dimensions are transposed
                        if (line_vuln_preds.shape[0] == vulnerable_lines.shape[0] and 
                            line_vuln_preds.shape[1] == vulnerable_lines.shape[2] and 
                            line_vuln_preds.shape[2] == vulnerable_lines.shape[1]):
                            # Transpose the targets to match predictions
                            vulnerable_lines_flat = vulnerable_lines.transpose(1, 2).contiguous()
                        else:
                            # Flatten both to [batch_size * seq_len * num_vuln_types]
                            line_vuln_preds = line_vuln_preds.view(-1)
                            vulnerable_lines_flat = vulnerable_lines.view(-1)
                    else:
                        vulnerable_lines_flat = vulnerable_lines
                    
                    # Calculate metrics with proper shape handling
                    correct_predictions = ((line_vuln_preds == vulnerable_lines_flat.float()) & (vulnerable_lines_flat.float() == 1)).sum().item()
                    total_vulnerable = vulnerable_lines_flat.sum().item()
                    predicted_vulnerable = line_vuln_preds.sum().item()
                    
                    if total_vulnerable > 0:
                        line_vuln_recall = correct_predictions / total_vulnerable
                    
                    if predicted_vulnerable > 0:
                        line_vuln_precision = correct_predictions / predicted_vulnerable
                    
                    total_predictions = line_vuln_preds.numel()
                    if total_predictions > 0:
                        line_vuln_accuracy = ((line_vuln_preds == vulnerable_lines_flat.float())).sum().item() / total_predictions
                    
                    # NEW: Debug metrics
                    if batch_idx == 0:
                        print(f"Correct predictions: {correct_predictions}")
                        print(f"Total vulnerable: {total_vulnerable}")
                        print(f"Predicted vulnerable: {predicted_vulnerable}")
                        print(f"Line accuracy: {line_vuln_accuracy:.4f}")
                        print(f"Line precision: {line_vuln_precision:.4f}")
                        print(f"Line recall: {line_vuln_recall:.4f}")
                        print("=" * 50)
                        
                except Exception as e:
                    print(f"Error calculating line metrics: {str(e)}")
                    # Set default values if calculation fails
                    line_vuln_accuracy = 0.0
                    line_vuln_precision = 0.0
                    line_vuln_recall = 0.0
                
                # Calculate batch vulnerability statistics
                batch_contract_vulns = contract_vulnerabilities.sum().item()
                batch_line_vulns = vulnerable_lines.sum().item()
                batch_size = contract_vulnerabilities.size(0)
                
                # IMPROVED: Balanced line vulnerability loss adjustment
                if batch_line_vulns > 0:
                    # If we have line vulnerabilities, use less aggressive focal loss
                    self.spatial_focal_loss.alpha = 0.1  # Lower alpha for stability
                    self.spatial_focal_loss.gamma = 1.5  # Lower gamma for stability
                    self.spatial_focal_loss.spatial_weight = 0.1  # Lower spatial weight
                else:
                    # If no line vulnerabilities, use very conservative settings
                    self.spatial_focal_loss.alpha = 0.05  # Very low alpha
                    self.spatial_focal_loss.gamma = 1.0  # Very low gamma
                    self.spatial_focal_loss.spatial_weight = 0.05  # Very low spatial weight
                
                # Apply minimum loss thresholds to prevent losses from going to zero
                contract_vuln_loss = torch.max(contract_vuln_loss, torch.tensor(0.0001).to(self.device))
                line_vuln_loss = torch.max(line_vuln_loss, torch.tensor(0.000001).to(self.device))  # Much smaller minimum for line loss
                
                # NEW: Add gradient scaling for line vulnerability loss to prevent gradient explosion
                if line_vuln_loss > 5.0:
                    line_vuln_loss = line_vuln_loss * 0.1  # Scale down high losses
                    print(f"⚠️  High line vulnerability loss detected: {line_vuln_loss.item():.6f}, scaling down")
                elif line_vuln_loss > 1.0:
                    line_vuln_loss = line_vuln_loss * 0.5  # Scale down moderately high losses
                    print(f"⚠️  Moderately high line vulnerability loss detected: {line_vuln_loss.item():.6f}, scaling down")
                
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
                
                # IMPROVED: Balanced loss with equal importance for all components
                # NEW: Apply warm-up for line vulnerability detection
                if self.use_augmentation and self.use_gan:
                    # Balanced weights for all components
                    total_loss = (
                        0.5 * gen_loss +  # Generation is important
                        0.25 * contract_vuln_loss * self.contract_vuln_weight +  # Contract-level detection
                        0.2 * line_vuln_loss * line_vuln_weight_adjusted +  # Line-level detection with warm-up
                        0.05 * discriminator_loss  # GAN component
                    )
                elif self.use_augmentation:
                    total_loss = (
                        0.6 * gen_loss +  # Generation is important
                        0.25 * contract_vuln_loss * self.contract_vuln_weight +  # Contract-level detection
                        0.15 * line_vuln_loss * line_vuln_weight_adjusted  # Line-level detection with warm-up
                    )
                else:
                    total_loss = (
                        0.5 * gen_loss +  # Generation is important
                        0.3 * contract_vuln_loss * self.contract_vuln_weight +  # Contract-level detection
                        0.2 * line_vuln_loss * line_vuln_weight_adjusted  # Line-level detection with warm-up
                    )
                
                # Add GAN losses to total loss
                if self.use_gan:
                    # Add adversarial loss if discriminator is too confident
                    if adversarial_loss > 0:
                        total_loss = total_loss + 0.02 * adversarial_loss  # Reduced from 0.05
                
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
                
                # IMPROVED: Clip vulnerability head gradients separately with higher limits
                vuln_params = [p for name, p in self.model.named_parameters() 
                             if 'vulnerability_head' in name or 'line_feature_extractor' in name or 
                                'line_vuln_attention' in name or 'vuln_type_attention' in name]
                if vuln_params:
                    torch.nn.utils.clip_grad_norm_(vuln_params, self.max_grad_norm * 2.0)  # Moderate limit for line vulnerability heads
                
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
                
                # Update progress bar with line-level metrics
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'contract_vuln_loss': f'{contract_vuln_loss.item():.4f}',
                    'line_vuln_loss': f'{line_vuln_loss.item():.6f}',  # More precision for line loss
                    'line_acc': f'{line_vuln_accuracy:.3f}',  # NEW: Line accuracy
                    'line_prec': f'{line_vuln_precision:.3f}',  # NEW: Line precision
                    'line_rec': f'{line_vuln_recall:.3f}',  # NEW: Line recall
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'grad_norm': f'{total_norm:.2f}',
                    'aug': 'ON' if self.use_augmentation else 'OFF',
                    'gan': 'ON' if self.use_gan else 'OFF',
                    'disc_loss': f'{discriminator_loss:.4f}' if self.use_gan else 'N/A',
                    'disc_conf': f'{discriminator_confidence:.3f}' if self.use_gan else 'N/A',
                    'line_weight': f'{self.line_vuln_weight:.0f}',  # Show line vulnerability weight
                    'contract_weight': f'{self.contract_vuln_weight:.0f}',  # Show contract vulnerability weight
                    'syntax': f'{syntax_penalty.item():.4f}' if syntax_penalty is not None else 'N/A',
                    'loss_scale': f'{self.line_loss_scale:.2f}',  # NEW: Show loss scale
                    'stab_factor': f'{self.stability_factor:.2f}'  # NEW: Show stability factor
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
        
        progress_bar.close()
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'contract_vuln_loss': total_contract_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'line_vuln_loss': total_line_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'discriminator_loss': total_discriminator_loss / batch_count if batch_count > 0 else 0.0,
            'syntax_loss': total_syntax_loss / batch_count if batch_count > 0 else 0.0,
            'line_vuln_accuracy': line_vuln_accuracy,  # NEW: Return line metrics
            'line_vuln_precision': line_vuln_precision,  # NEW: Return line metrics
            'line_vuln_recall': line_vuln_recall  # NEW: Return line metrics
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
                    
                    # NEW: Calculate warm-up factor early in the training loop
                    warmup_factor = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
                    line_vuln_weight_adjusted = self.line_vuln_weight * warmup_factor
                    
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
                        gen_loss = self.generator_loss_fn(logits, target_ids_shifted, None)
                    except KeyError as e:
                        print(f"Missing key in outputs: {e}")
                        print(f"Available keys: {list(outputs.keys())}")
                        continue
                    except Exception as e:
                        print(f"Error calculating generation loss: {str(e)}")
                        continue
                    
                    # Calculate contract-level vulnerability loss
                    contract_vuln_loss = self.contract_focal_loss(
                        contract_vuln_logits,
                        contract_vulnerabilities.float()
                    )
                    
                    # IMPROVED: Calculate line-level vulnerability loss with spatial awareness
                    # FIXED: Handle transposed dimensions for vulnerable_lines
                    if line_vuln_logits.shape != vulnerable_lines.shape:
                        # Check if dimensions are transposed and fix them
                        if (line_vuln_logits.shape[0] == vulnerable_lines.shape[0] and 
                            line_vuln_logits.shape[1] == vulnerable_lines.shape[2] and 
                            line_vuln_logits.shape[2] == vulnerable_lines.shape[1]):
                            # Transpose vulnerable_lines to match line_vuln_logits
                            vulnerable_lines_for_loss = vulnerable_lines.transpose(1, 2).contiguous()
                        else:
                            vulnerable_lines_for_loss = vulnerable_lines
                    else:
                        vulnerable_lines_for_loss = vulnerable_lines
                    
                    line_vuln_loss = self.spatial_focal_loss(
                        line_vuln_logits.view(-1, self.model.num_vulnerability_types),
                        vulnerable_lines_for_loss.view(-1, self.model.num_vulnerability_types).float(),
                        token_to_line.view(-1) if token_to_line is not None else None
                    )
                    
                    # Apply minimum loss thresholds to prevent losses from going to zero
                    contract_vuln_loss = torch.max(contract_vuln_loss, torch.tensor(0.0001).to(self.device))
                    line_vuln_loss = torch.max(line_vuln_loss, torch.tensor(0.00001).to(self.device))  # Increased from 0.01
                    
                    # IMPROVED: Combined loss with balanced weights
                    if self.use_augmentation:
                        total_loss = (
                            0.6 * gen_loss +
                            0.25 * contract_vuln_loss * self.contract_vuln_weight +  # Use contract vulnerability weight
                            0.15 * line_vuln_loss * self.line_vuln_weight  # Use dynamic weight
                        )
                    else:
                        total_loss = (
                            0.5 * gen_loss + 
                            0.3 * contract_vuln_loss * self.contract_vuln_weight +  # Use contract vulnerability weight
                            0.2 * line_vuln_loss * self.line_vuln_weight  # Use dynamic weight
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
            
            # Update current epoch for warm-up
            self.current_epoch = epoch
            
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
            
            # NEW: Track line-level performance metrics
            self.training_history['line_vuln_accuracy'].append(train_metrics.get('line_vuln_accuracy', 0.0))
            self.training_history['line_vuln_precision'].append(train_metrics.get('line_vuln_precision', 0.0))
            self.training_history['line_vuln_recall'].append(train_metrics.get('line_vuln_recall', 0.0))
            
            if self.use_gan:
                self.training_history['discriminator_loss'].append(train_metrics['discriminator_loss'])
            
            # NEW: Track syntax loss
            self.training_history['syntax_loss'].append(train_metrics.get('syntax_loss', 0.0))
            
            # Print metrics with line-level performance
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Contract Vulnerability Loss: {train_metrics['contract_vuln_loss']:.4f}")
            print(f"Line Vulnerability Loss: {train_metrics['line_vuln_loss']:.6f}")
            print(f"Line Accuracy: {train_metrics.get('line_vuln_accuracy', 0.0):.4f}")  # NEW
            print(f"Line Precision: {train_metrics.get('line_vuln_precision', 0.0):.4f}")  # NEW
            print(f"Line Recall: {train_metrics.get('line_vuln_recall', 0.0):.4f}")  # NEW
            if self.use_gan:
                print(f"Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
            print(f"Syntax Loss: {train_metrics.get('syntax_loss', 0.0):.4f}")  # NEW: Show syntax loss
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # NEW: Check line-level performance and adjust training if needed
            line_recall = train_metrics.get('line_vuln_recall', 0.0)
            line_precision = train_metrics.get('line_vuln_precision', 0.0)
            
            # NEW: Reset oscillation detection if model stabilizes
            if not self.oscillation_detected and epoch > 2:
                # If no oscillation for 2 epochs, gradually increase loss scale
                if self.line_loss_scale < 1.0:
                    self.line_loss_scale = min(1.0, self.line_loss_scale * 1.2)
                    print(f"Model stabilized. Increasing line loss scale to: {self.line_loss_scale:.3f}")
                
                # Also increase stability factor
                if self.stability_factor < 1.0:
                    self.stability_factor = min(1.0, self.stability_factor * 1.1)
                    print(f"Increasing stability factor to: {self.stability_factor:.3f}")
            
            # Reset oscillation detection for next epoch
            self.oscillation_detected = False
            
            # NEW: Stability adjustment to prevent oscillation
            if epoch > 0:  # After first epoch
                # Check for oscillation between extremes
                if (self.prev_line_recall > 0.8 and line_recall < 0.1) or (self.prev_line_recall < 0.1 and line_recall > 0.8):
                    print(f"⚠️  Oscillation detected! Recall: {self.prev_line_recall:.3f} → {line_recall:.3f}")
                    print("Reducing line vulnerability loss weight to stabilize training...")
                    self.stability_factor = max(0.3, self.stability_factor * 0.7)  # Reduce by 30%
                    print(f"Stability factor: {self.stability_factor:.3f}")
                
                # Check for extreme precision/recall imbalance
                if line_precision < 0.01 and line_recall > 0.8:
                    print(f"⚠️  High recall ({line_recall:.3f}) but very low precision ({line_precision:.3f})")
                    print("Adjusting loss to improve precision...")
                    self.spatial_focal_loss.alpha = min(0.5, self.spatial_focal_loss.alpha * 1.2)
                    self.spatial_focal_loss.gamma = max(1.5, self.spatial_focal_loss.gamma * 0.9)
                
                if line_precision > 0.8 and line_recall < 0.1:
                    print(f"⚠️  High precision ({line_precision:.3f}) but very low recall ({line_recall:.3f})")
                    print("Adjusting loss to improve recall...")
                    self.spatial_focal_loss.alpha = max(0.1, self.spatial_focal_loss.alpha * 0.8)
                    self.spatial_focal_loss.gamma = min(3.0, self.spatial_focal_loss.gamma * 1.1)
            
            # Update previous values for next epoch
            self.prev_line_recall = line_recall
            self.prev_line_precision = line_precision
            
            if line_recall < 0.01 and epoch > 5:  # If line recall is very low after 5 epochs
                print(f"⚠️  Very low line recall detected: {line_recall:.4f}")
                print("Boosting line vulnerability learning rate...")
                
                # Boost learning rate for line vulnerability heads
                for param_group in self.optimizer.param_groups:
                    if any(name in str(param_group) for name in ['line_vulnerability_head', 'spatial_attention', 'line_feature_extractor']):
                        param_group['lr'] = param_group['lr'] * 2.0
                        print(f"Boosted line head LR to: {param_group['lr']:.6f}")
            
            # NEW: Check if model is consistently making no predictions
            if line_recall == 0.0 and epoch > 5:  # Increased from 3 to 5
                print(f"⚠️  Model making no line vulnerability predictions!")
                print("Applying conservative learning rate boost...")
                
                # Boost learning rate conservatively for line vulnerability heads
                for param_group in self.optimizer.param_groups:
                    if any(name in str(param_group) for name in ['line_vulnerability_head', 'line_feature_extractor']):
                        param_group['lr'] = param_group['lr'] * 2.0  # Reduced from 5.0 to 2.0
                        print(f"Conservatively boosted line head LR to: {param_group['lr']:.6f}")
                
                # Also increase loss scale conservatively
                self.line_loss_scale = min(self.max_line_loss_scale, self.line_loss_scale * 1.5)  # Reduced from 3.0 to 1.5
                print(f"Conservatively increased line loss scale to: {self.line_loss_scale:.3f}")
                
                # Reset stability factor conservatively
                self.stability_factor = min(1.0, self.stability_factor * 1.2)  # Reduced from 2.0 to 1.2
                print(f"Reset stability factor to: {self.stability_factor:.3f}")
            
            # NEW: Check prediction tracking for more conservative intervention
            if self.batches_with_predictions == 0 and epoch > 5:  # Increased from 2 to 5
                print(f"🚨  NO PREDICTIONS IN ANY BATCH!")
                print(f"Total predictions this epoch: {self.total_line_predictions}")
                print("Applying conservative emergency intervention...")
                
                # Conservative learning rate boost
                for param_group in self.optimizer.param_groups:
                    if any(name in str(param_group) for name in ['line_vulnerability_head', 'line_feature_extractor']):
                        param_group['lr'] = param_group['lr'] * 3.0  # Reduced from 10.0 to 3.0
                        print(f"Conservative emergency boosted line head LR to: {param_group['lr']:.6f}")
                
                # Conservative loss scale
                self.line_loss_scale = min(self.max_line_loss_scale, self.line_loss_scale * 2.0)  # Reduced from max to 2.0
                print(f"Set line loss scale to conservative maximum: {self.line_loss_scale:.3f}")
                
                # Reset stability factors conservatively
                self.stability_factor = 0.8  # Reduced from 1.0 to 0.8
                print(f"Reset stability factor to: {self.stability_factor:.3f}")
            
            # Reset prediction tracking for next epoch
            self.total_line_predictions = 0
            self.batches_with_predictions = 0
            
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
                    'use_gan': self.use_gan,
                    'line_vuln_accuracy': train_metrics.get('line_vuln_accuracy', 0.0),  # NEW: Save line metrics
                    'line_vuln_precision': train_metrics.get('line_vuln_precision', 0.0),  # NEW: Save line metrics
                    'line_vuln_recall': train_metrics.get('line_vuln_recall', 0.0)  # NEW: Save line metrics
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"🎉 New best validation loss! Saved checkpoint to {checkpoint_path}")
                print(f"Line-level performance: Acc={train_metrics.get('line_vuln_accuracy', 0.0):.4f}, "
                      f"Prec={train_metrics.get('line_vuln_precision', 0.0):.4f}, "
                      f"Rec={train_metrics.get('line_vuln_recall', 0.0):.4f}")
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
                'use_gan': self.use_gan,
                'line_vuln_accuracy': train_metrics.get('line_vuln_accuracy', 0.0),  # NEW: Save line metrics
                'line_vuln_precision': train_metrics.get('line_vuln_precision', 0.0),  # NEW: Save line metrics
                'line_vuln_recall': train_metrics.get('line_vuln_recall', 0.0)  # NEW: Save line metrics
            }
            
            torch.save(checkpoint_data, latest_checkpoint_path)