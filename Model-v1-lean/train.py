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

class PathAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, context=None):
        if context is None:
            return x
        attn_output, _ = self.attention(x, context, context)
        return self.norm(x + attn_output)

class GrammarConstraint(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        return self.norm(x)

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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)
        log_prob = F.log_softmax(pred, dim=-1)
        loss = (-smooth_one_hot * log_prob).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class Discriminator(nn.Module):
    def __init__(
        self,
        d_model=768,
        dropout=0.2,
        use_layer_norm=True,
        vocab_size=50265
    ):
        super().__init__()
        
        # Path-aware attention mechanism
        self.path_attention = PathAwareAttention(d_model)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Grammar constraint
        self.grammar_constraint = GrammarConstraint(vocab_size, d_model)
        
        # Separate heads for vulnerability and synthetic detection
        self.vulnerability_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary classification for vulnerability
        )
        
        self.synthetic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary classification for synthetic detection
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, d_model]
        Returns:
            vulnerability_logits: Logits for vulnerability detection
            synthetic_logits: Logits for synthetic detection
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions from both heads
        vulnerability_logits = self.vulnerability_head(features)
        synthetic_logits = self.synthetic_head(features)
        
        return vulnerability_logits, synthetic_logits

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
        
        # State variable declarations
        self.state_variable_types = [
            'uint', 'int', 'bool', 'string', 'bytes',
            'address', 'mapping', 'array'
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
    
    def apply_constraints(self, logits, previous_tokens):
        """Apply Solidity-specific constraints to logits"""
        if len(previous_tokens) == 0:
            return logits
        
        # Initialize balance stack for this sample
        balance_stack = []
        
        # Get the last token
        last_token = previous_tokens[-1]
        last_token_str = self.tokenizer.decode([last_token])
        
        # Apply keyword constraints
        if last_token_str in self.keyword_constraints:
            allowed_tokens = self.keyword_constraints[last_token_str]
            allowed_indices = []
            for token in allowed_tokens:
                token_idx = self.tokenizer.convert_tokens_to_ids(token)
                if token_idx != self.tokenizer.unk_token_id:  # Skip unknown tokens
                    allowed_indices.append(token_idx)
            
            if allowed_indices:  # Only apply mask if we have valid indices
                mask = torch.zeros_like(logits)
                mask[allowed_indices] = 1
                logits = logits.masked_fill(mask == 0, float('-inf'))
        
        # Apply balanced pair constraints
        for open_token, close_token in self.balanced_pairs:
            open_idx = self.tokenizer.convert_tokens_to_ids(open_token)
            close_idx = self.tokenizer.convert_tokens_to_ids(close_token)
            
            if open_idx != self.tokenizer.unk_token_id and close_idx != self.tokenizer.unk_token_id:
                if last_token == open_idx:
                    balance_stack.append(close_idx)
                elif last_token == close_idx and balance_stack:
                    balance_stack.pop()
        
        # Prevent closing tokens if no matching open token
        if not balance_stack:
            for _, close_token in self.balanced_pairs:
                close_idx = self.tokenizer.convert_tokens_to_ids(close_token)
                if close_idx != self.tokenizer.unk_token_id:
                    logits[close_idx] = float('-inf')
        
        # Apply semicolon constraints
        if last_token_str in self.semicolon_required:
            semicolon_idx = self.tokenizer.convert_tokens_to_ids(';')
            if semicolon_idx != self.tokenizer.unk_token_id:
                logits[semicolon_idx] = logits[semicolon_idx] * 2  # Increase probability of semicolon
        
        return logits
    
    def compute_grammar_loss(self, logits, target_ids):
        """Compute grammar-related loss"""
        # Simple implementation - can be enhanced with more sophisticated grammar rules
        return torch.tensor(0.0, device=logits.device)
    
    def compute_validity_loss(self, logits, target_ids):
        """Compute validity-related loss"""
        # Simple implementation - can be enhanced with more sophisticated validity checks
        return torch.tensor(0.0, device=logits.device)

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
        pad_token_id=0,
        d_model=768,
        gpu_id=1,
        gradient_accumulation_steps=4,
        generation_top_p=0.9,
        generation_temperature=0.8,
        generation_top_k=40,
        min_generation_length=50
    ):
        # Set memory optimization settings
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model = model.to(self.device)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gpu_id = gpu_id
        
        # Store tokenizer
        self.tokenizer = tokenizer
        
        # Initialize SolidityTokenConstraints with tokenizer
        self.token_constraints = SolidityTokenConstraints(tokenizer)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Generation parameters
        self.generation_top_p = generation_top_p
        self.generation_temperature = generation_temperature
        self.generation_top_k = generation_top_k
        self.min_generation_length = min_generation_length
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            d_model=d_model,
            dropout=0.2,
            use_layer_norm=True,
            vocab_size=tokenizer.vocab_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=100,  # Will be updated in train()
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1,  # 10% warmup
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        self.max_grad_norm = max_grad_norm
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize focal loss
        self.focal_loss = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction='mean'
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        
        total_gen_loss = 0
        total_vuln_loss = 0
        batch_count = 0
        
        # Initialize progress bar with total steps
        total_steps = len(self.train_dataloader)
        progress_bar = tqdm(
            total=total_steps,
            desc=f'Epoch {epoch}',
            bar_format='{l_bar}{bar:10}{r_bar}',
            dynamic_ncols=True
        )
        
        for step, batch in enumerate(self.train_dataloader):
            try:
                # Move tensors to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                path_input_ids = batch['path_input_ids']
                path_attention_mask = batch['path_attention_mask']
                target_ids = batch['target_ids']
                is_vulnerable = batch['label'].float().view(-1, 1)
                
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
                    
                    # Get logits and apply token constraints
                    logits = outputs['logits']
                    processed_target_ids = outputs['target_ids']
                    
                    # Vectorized token constraints application
                    batch_size = input_ids.size(0)
                    seq_len = logits.size(0) // batch_size  # Use actual sequence length from logits
                    vocab_size = logits.size(-1)
                    
                    # Reshape logits to [batch_size, seq_len, vocab_size]
                    logits = logits.contiguous().view(batch_size, seq_len, vocab_size)
                    
                    # Apply constraints in parallel for each position
                    for pos in range(seq_len):
                        # Get all previous tokens up to current position
                        prev_tokens = input_ids[:, :pos+1]
                        # Apply constraints for all samples in batch at once
                        logits[:, pos] = self.token_constraints.apply_constraints_batch(
                            logits[:, pos],
                            prev_tokens
                        )
                    
                    # Reshape back to [batch_size * seq_len, vocab_size]
                    logits = logits.view(-1, vocab_size)
                    
                    # Train discriminator on real data
                    encoder_output = outputs['encoder_output']
                    vuln_pred, _ = self.discriminator(encoder_output.detach())
                    vuln_loss_real = self.focal_loss(vuln_pred, is_vulnerable)
                    
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
                    vuln_pred_synth, _ = self.discriminator(synthetic_encoder_output)
                    vuln_loss_synth = self.focal_loss(vuln_pred_synth, torch.zeros_like(vuln_pred_synth))
                    
                    # Combined discriminator loss
                    d_loss = (vuln_loss_real + vuln_loss_synth) / 2
                
                # Backward pass for discriminator with gradient scaling
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.discriminator_optimizer)
                self.scaler.update()
                
                # Train generator
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # Compute generator loss
                    gen_loss = F.cross_entropy(
                        logits,
                        processed_target_ids.view(-1)
                    )
                    
                    # Adversarial loss for generator
                    vuln_pred, _ = self.discriminator(synthetic_encoder_output)
                    adv_vuln_loss = self.focal_loss(vuln_pred, torch.zeros_like(vuln_pred))
                    
                    # Combined generator loss
                    total_loss = gen_loss + 0.1 * adv_vuln_loss
                
                # Backward pass for generator with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update metrics
                total_gen_loss += gen_loss.item()
                total_vuln_loss += (vuln_loss_real.item() + vuln_loss_synth.item()) / 2
                batch_count += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'vuln_loss': f'{vuln_loss_real.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Clear memory more efficiently
                del outputs, synthetic_outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                print(f"Error occurred at step {step}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate and print epoch averages
        avg_metrics = {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf')
        }
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Generator Loss: {avg_metrics['gen_loss']:.4f}")
        print(f"Average Vulnerability Loss: {avg_metrics['vuln_loss']:.4f}")
        
        return avg_metrics

    def validate(self):
        self.model.eval()
        self.discriminator.eval()
        
        total_gen_loss = 0
        total_vuln_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Move tensors to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    path_input_ids = batch['path_input_ids']
                    path_attention_mask = batch['path_attention_mask']
                    target_ids = batch['target_ids']
                    is_vulnerable = batch['label'].float().view(-1, 1)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        path_input_ids=path_input_ids,
                        path_attention_mask=path_attention_mask,
                        target_ids=target_ids
                    )
                    
                    # Get logits and apply token constraints
                    logits = outputs['logits']
                    processed_target_ids = outputs['target_ids']
                    
                    # Apply token constraints
                    batch_size = input_ids.size(0)
                    seq_len = input_ids.size(1)
                    
                    for i in range(batch_size):
                        start_idx = i * seq_len
                        end_idx = (i + 1) * seq_len
                        sample_logits = logits[start_idx:end_idx]
                        sample_input_ids = input_ids[i]
                        actual_seq_len = sample_logits.size(0)
                        
                        for j in range(actual_seq_len):
                            sample_logits[j] = self.token_constraints.apply_constraints(
                                sample_logits[j],
                                sample_input_ids[:j+1].tolist()
                            )
                    
                    # Calculate generator loss
                    gen_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        processed_target_ids.view(-1)
                    )
                    
                    # Get discriminator predictions
                    vuln_pred, _ = self.discriminator(outputs['encoder_output'])
                    vuln_loss = self.focal_loss(vuln_pred, is_vulnerable)
                    
                    # Update metrics
                    total_gen_loss += gen_loss.item()
                    total_vuln_loss += vuln_loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf')
        }

    def train(self, num_epochs, checkpoint_dir='checkpoints'):
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Update scheduler epochs
        self.scheduler.epochs = num_epochs
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['gen_loss'])
            self.training_history['val_loss'].append(val_metrics['gen_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Vulnerability Loss: {train_metrics['vuln_loss']:.4f}")
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
                print(f"🎉 New best validation loss! Saved checkpoint to {checkpoint_path}")
            
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
