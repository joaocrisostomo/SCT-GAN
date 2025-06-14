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
    
    def _init_constraint_masks(self):
        """Initialize masks for token constraints"""
        self.keyword_mask = torch.zeros(self.tokenizer.vocab_size)
        self.balance_mask = torch.zeros(self.tokenizer.vocab_size)
        self.semicolon_mask = torch.zeros(self.tokenizer.vocab_size)
        
        # Convert token constraints to indices
        for keyword, allowed_tokens in self.keyword_constraints.items():
            keyword_idx = self.tokenizer.convert_tokens_to_ids(keyword)
            if keyword_idx != self.tokenizer.unk_token_id:  # Skip unknown tokens
                for token in allowed_tokens:
                    token_idx = self.tokenizer.convert_tokens_to_ids(token)
                    if token_idx != self.tokenizer.unk_token_id:  # Skip unknown tokens
                        self.keyword_mask[token_idx] = 1
    
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
        gpu_id=0,
        gradient_accumulation_steps=4,
        generation_top_p=0.9,
        generation_temperature=0.8,
        generation_top_k=40,
        min_generation_length=50,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        label_smoothing=0.1
    ):
        # Set memory optimization settings
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set environment variable for memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model = model.to(self.device)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gpu_id = gpu_id
        
        # Store tokenizer
        self.tokenizer = tokenizer
        
        # Initialize SolidityTokenConstraints with tokenizer
        self.token_constraints = SolidityTokenConstraints(tokenizer)
        
        # Initialize PathAwareAttention
        self.path_attention = PathAwareAttention(d_model).to(self.device)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Generation parameters
        self.generation_top_p = generation_top_p
        self.generation_temperature = generation_temperature
        self.generation_top_k = generation_top_k
        self.min_generation_length = min_generation_length
        
        # Loss parameters
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = label_smoothing
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            d_model=d_model,
            dropout=0.2,
            use_layer_norm=True,
            vocab_size=tokenizer.vocab_size
        ).to(self.device)
        
        # Initialize optimizer with improved settings
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.path_attention.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
            amsgrad=True
        )
        
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
            amsgrad=True
        )
        
        # Learning rate scheduler with warmup
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
            'learning_rate': [],
            'vulnerability_loss': [],
            'synthetic_loss': [],
            'grammar_loss': [],
            'validity_loss': [],
            'generation_quality': []
        }
        
        # Mixed precision with improved settings
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2.**12,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100
        )
        
        # Initialize focal loss
        self.focal_loss = FocalLoss(
            alpha=focal_loss_alpha,
            gamma=focal_loss_gamma,
            reduction='mean'
        )
        
        # Initialize label smoothing
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            reduction='mean'
        )
        
        # Add penalty parameters
        self.lambda_d = 0.1  # Penalty strength for discriminator
        self.lambda_g = 0.1  # Penalty strength for generator
        self.delta_d = 0.2   # Threshold for discriminator loss
        self.delta_g = 0.2   # Threshold for generator loss

    def compute_penalty(self, loss, lambda_param, delta):
        """
        Compute penalty term to prevent mode collapse and encourage diversity.
        
        Args:
            loss: The current loss value
            lambda_param: Penalty strength parameter
            delta: Threshold for applying penalty
            
        Returns:
            penalty: The computed penalty term
        """
        # Apply penalty if loss is below threshold
        if loss < delta:
            return lambda_param * (delta - loss)
        return torch.tensor(0.0, device=loss.device)

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        self.path_attention.train()
        
        total_gen_loss = 0
        total_vuln_loss = 0
        total_synth_loss = 0
        total_grammar_loss = 0
        total_validity_loss = 0
        total_generation_quality = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        for step, batch in enumerate(progress_bar):
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
                    
                    # Get encoder output and apply path-aware attention
                    encoder_output = outputs['encoder_output']
                    path_features = outputs.get('path_features', None)
                    if path_features is not None:
                        encoder_output = self.path_attention(encoder_output, path_features)
                    
                    # Get logits and apply token constraints
                    logits = outputs['logits']
                    processed_target_ids = outputs['target_ids']
                    
                    # Apply token constraints directly on flattened logits
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
                    
                    # Train discriminator on real data
                    vuln_pred, synth_pred = self.discriminator(encoder_output.detach())
                    vuln_loss_real = self.focal_loss(vuln_pred, is_vulnerable)
                    
                    # For synthetic detection, randomly choose between real and synthetic
                    if torch.rand(1) < 0.5:
                        synth_loss_real = self.focal_loss(synth_pred, torch.zeros_like(synth_pred))
                        is_synthetic = torch.zeros_like(synth_pred)
                    else:
                        synth_loss_real = self.focal_loss(synth_pred, torch.ones_like(synth_pred))
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
                        if path_features is not None:
                            synthetic_encoder_output = self.path_attention(synthetic_encoder_output, path_features)
                    
                    # Train discriminator on synthetic data
                    vuln_pred_synth, synth_pred_synth = self.discriminator(synthetic_encoder_output)
                    vuln_loss_synth = self.focal_loss(vuln_pred_synth, torch.zeros_like(vuln_pred_synth))
                    synth_loss_synth = self.focal_loss(synth_pred_synth, torch.ones_like(synth_pred_synth))
                    
                    # Combined discriminator loss
                    d_loss = (vuln_loss_real + synth_loss_real + vuln_loss_synth + synth_loss_synth) / 2
                    d_penalty = self.compute_penalty(d_loss, self.lambda_d, self.delta_d)
                    d_loss = d_loss + d_penalty
                
                # Backward pass for discriminator with gradient scaling
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.discriminator_optimizer)
                self.scaler.update()
                
                # Train generator
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # Compute generator loss with label smoothing
                    gen_loss = self.label_smoothing_loss(
                        logits.view(-1, logits.size(-1)),
                        processed_target_ids.view(-1)
                    )
                    
                    # Calculate grammar and validity losses
                    grammar_loss = self.token_constraints.compute_grammar_loss(logits, processed_target_ids)
                    validity_loss = self.token_constraints.compute_validity_loss(logits, processed_target_ids)
                    
                    # Adversarial loss for generator
                    vuln_pred, synth_pred = self.discriminator(synthetic_encoder_output)
                    adv_vuln_loss = self.focal_loss(vuln_pred, torch.zeros_like(vuln_pred))
                    adv_synth_loss = self.focal_loss(synth_pred, torch.zeros_like(synth_pred))
                    
                    # Calculate generation quality
                    generation_quality = torch.mean(torch.abs(synthetic_encoder_output - encoder_output.detach()))
                    
                    # Combined generator loss
                    g_loss = (
                        gen_loss + 
                        0.1 * (adv_vuln_loss + adv_synth_loss) + 
                        0.05 * grammar_loss + 
                        0.05 * validity_loss - 
                        0.01 * generation_quality  # Encourage diversity
                    )
                    g_penalty = self.compute_penalty(g_loss, self.lambda_g, self.delta_g)
                    total_loss = g_loss + g_penalty
                
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
                total_synth_loss += (synth_loss_real.item() + synth_loss_synth.item()) / 2
                total_grammar_loss += grammar_loss.item()
                total_validity_loss += validity_loss.item()
                total_generation_quality += generation_quality.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'vuln_loss': f'{vuln_loss_real.item():.4f}',
                    'synth_loss': f'{synth_loss_real.item():.4f}',
                    'grammar_loss': f'{grammar_loss.item():.4f}',
                    'validity_loss': f'{validity_loss.item():.4f}',
                    'gen_quality': f'{generation_quality.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Clear memory
                del outputs, synthetic_outputs, encoder_output, synthetic_encoder_output
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                print(f"Error occurred at step {step}")
                import traceback
                print(traceback.format_exc())
                continue
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'synth_loss': total_synth_loss / batch_count if batch_count > 0 else float('inf'),
            'grammar_loss': total_grammar_loss / batch_count if batch_count > 0 else float('inf'),
            'validity_loss': total_validity_loss / batch_count if batch_count > 0 else float('inf'),
            'generation_quality': total_generation_quality / batch_count if batch_count > 0 else float('inf')
        }
    
    def validate(self):
        self.model.eval()
        self.discriminator.eval()
        self.path_attention.eval()
        
        total_gen_loss = 0
        total_vuln_loss = 0
        total_synth_loss = 0
        total_grammar_loss = 0
        total_validity_loss = 0
        total_generation_quality = 0
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
                    
                    # Get encoder output and apply path-aware attention
                    encoder_output = outputs['encoder_output']
                    path_features = outputs.get('path_features', None)
                    if path_features is not None:
                        encoder_output = self.path_attention(encoder_output, path_features)
                    
                    # Get logits and apply token constraints
                    logits = outputs['logits']
                    processed_target_ids = outputs['target_ids']
                    
                    # Apply token constraints
                    for i in range(logits.size(0)):
                        logits[i] = self.token_constraints.apply_constraints(
                            logits[i],
                            input_ids[i].tolist()
                        )
                    
                    # Calculate generator loss with label smoothing
                    gen_loss = self.label_smoothing_loss(
                        logits.view(-1, logits.size(-1)),
                        processed_target_ids.view(-1)
                    )
                    
                    # Get discriminator predictions
                    vuln_pred, synth_pred = self.discriminator(encoder_output)
                    vuln_loss = self.focal_loss(vuln_pred, is_vulnerable)
                    
                    # For synthetic detection, use a mix of real and synthetic data
                    if torch.rand(1) < 0.5:
                        synth_loss = self.focal_loss(synth_pred, torch.zeros_like(synth_pred))
                    else:
                        synth_loss = self.focal_loss(synth_pred, torch.ones_like(synth_pred))
                    
                    # Generate synthetic data for quality evaluation
                    synthetic_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        path_input_ids=path_input_ids,
                        path_attention_mask=path_attention_mask,
                        target_ids=None
                    )
                    synthetic_encoder_output = synthetic_outputs['encoder_output']
                    if path_features is not None:
                        synthetic_encoder_output = self.path_attention(synthetic_encoder_output, path_features)
                    
                    # Calculate generation quality
                    generation_quality = torch.mean(torch.abs(synthetic_encoder_output - encoder_output))
                    
                    # Calculate grammar and validity losses
                    grammar_loss = self.token_constraints.compute_grammar_loss(logits, processed_target_ids)
                    validity_loss = self.token_constraints.compute_validity_loss(logits, processed_target_ids)
                    
                    # Update metrics
                    total_gen_loss += gen_loss.item()
                    total_vuln_loss += vuln_loss.item()
                    total_synth_loss += synth_loss.item()
                    total_grammar_loss += grammar_loss.item()
                    total_validity_loss += validity_loss.item()
                    total_generation_quality += generation_quality.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue
        
        return {
            'gen_loss': total_gen_loss / batch_count if batch_count > 0 else float('inf'),
            'vuln_loss': total_vuln_loss / batch_count if batch_count > 0 else float('inf'),
            'synth_loss': total_synth_loss / batch_count if batch_count > 0 else float('inf'),
            'grammar_loss': total_grammar_loss / batch_count if batch_count > 0 else float('inf'),
            'validity_loss': total_validity_loss / batch_count if batch_count > 0 else float('inf'),
            'generation_quality': total_generation_quality / batch_count if batch_count > 0 else float('inf')
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
            self.training_history['vulnerability_loss'].append(train_metrics['vuln_loss'])
            self.training_history['synthetic_loss'].append(train_metrics['synth_loss'])
            self.training_history['grammar_loss'].append(train_metrics['grammar_loss'])
            self.training_history['validity_loss'].append(train_metrics['validity_loss'])
            self.training_history['generation_quality'].append(train_metrics['generation_quality'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['gen_loss']:.4f}")
            print(f"Val Loss: {val_metrics['gen_loss']:.4f}")
            print(f"Vulnerability Loss: {train_metrics['vuln_loss']:.4f}")
            print(f"Synthetic Loss: {train_metrics['synth_loss']:.4f}")
            print(f"Grammar Loss: {train_metrics['grammar_loss']:.4f}")
            print(f"Validity Loss: {train_metrics['validity_loss']:.4f}")
            print(f"Generation Quality: {train_metrics['generation_quality']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if validation loss improved
            if val_metrics['gen_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['gen_loss']
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'path_attention_state_dict': self.path_attention.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                    'val_loss': val_metrics['gen_loss'],
                    'training_history': self.training_history,
                    'generation_params': {
                        'top_p': self.generation_top_p,
                        'temperature': self.generation_temperature,
                        'top_k': self.generation_top_k,
                        'min_length': self.min_generation_length
                    }
                }, checkpoint_path)
                print(f"🎉 New best validation loss! Saved checkpoint to {checkpoint_path}")
            
            # Save latest checkpoint
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'path_attention_state_dict': self.path_attention.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                'val_loss': val_metrics['gen_loss'],
                'training_history': self.training_history,
                'generation_params': {
                    'top_p': self.generation_top_p,
                    'temperature': self.generation_temperature,
                    'top_k': self.generation_top_k,
                    'min_length': self.min_generation_length
                }
            }, latest_checkpoint_path)

    def generate_contract(self, input_ids, attention_mask, path_input_ids, path_attention_mask):
        """
        Generate a smart contract using improved token generation with nucleus sampling
        """
        self.model.eval()
        self.path_attention.eval()
        
        with torch.no_grad():
            # Get initial features
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                path_input_ids=path_input_ids,
                path_attention_mask=path_attention_mask,
                target_ids=None
            )
            
            features = outputs['encoder_output']
            context = outputs.get('path_features', None)
            
            # Initialize generation
            batch_size = input_ids.size(0)
            device = input_ids.device
            max_length = 1024  # Increased maximum contract length
            generated_tokens = []
            
            # Start with special tokens
            current_tokens = torch.tensor([[self.tokenizer.cls_token_id]], device=device)
            
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(
                    input_ids=current_tokens,
                    attention_mask=torch.ones_like(current_tokens),
                    path_input_ids=path_input_ids,
                    path_attention_mask=path_attention_mask,
                    target_ids=None
                )
                
                # Get logits and apply path-aware attention
                logits = outputs['logits'][:, -1, :]  # Get last token predictions
                logits = self.path_attention(logits, context)
                
                # Apply token constraints
                logits = self.token_constraints.apply_constraints(
                    logits,
                    current_tokens[0].tolist()
                )
                
                # Apply temperature scaling
                logits = logits / self.generation_temperature
                
                # Apply top-k filtering
                if self.generation_top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, self.generation_top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply nucleus sampling (top-p)
                if self.generation_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > self.generation_top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from logits
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append generated token
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Stop if we generate the end token and have reached minimum length
                if next_token.item() == self.tokenizer.sep_token_id and len(generated_tokens) >= self.min_generation_length:
                    break
            
            return torch.tensor(generated_tokens, device=device)
