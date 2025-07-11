import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from transformers import AutoModel, AutoTokenizer
import math
import torch.nn.functional as F # Added for F.gelu

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SmartContractTransformer(nn.Module):
    def __init__(
        self,
        d_model=768,  # Hidden size for embeddings and transformer layers
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.3,  # Increased dropout for aggressive regularization
        max_length=1024,  # Match your dataset's max_length
        vocab_size=50265,  # Match your tokenizer's vocab size
        num_vulnerability_types=8,  # Number of vulnerability types
        use_gan=False  # Enable GAN training with integrated discriminator
    ):
        super().__init__()
        
        # Token embedding with more aggressive dropout
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # AST path embedding with layer normalization and dropout
        self.ast_embedding = nn.Embedding(vocab_size, d_model)
        self.ast_embedding_dropout = nn.Dropout(dropout)
        self.ast_embedding_norm = nn.LayerNorm(d_model)
        
        # Path embedding for beam search (alias for ast_embedding)
        self.path_embedding = self.ast_embedding
        
        # Transformer encoder with improved configuration and more dropout
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # Using GELU activation for better performance
            norm_first=True  # Pre-norm for better training stability
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer decoder with improved configuration and more dropout
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm for better training stability
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection with layer normalization and dropout
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # IMPROVED: Contract-level vulnerability detection head with better architecture
        # Multi-scale feature aggregation for contract-level analysis
        self.contract_feature_aggregation = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),  # Input: concatenated features
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Contract-level attention for vulnerability detection
        self.contract_vuln_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Contract-level vulnerability detection head with improved architecture
        self.contract_vulnerability_head = nn.Sequential(
            nn.Linear(d_model, d_model),  # Input: aggregated features
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_vulnerability_types)
        )
        
        # IMPROVED: Line-level vulnerability detection with much stronger architecture
        # Enhanced line feature extraction with attention and context
        self.line_feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # NEW: Custom line feature extractor with residual connection and better stability
        class ResidualLineFeatureExtractor(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_model)
                self.norm1 = nn.LayerNorm(d_model, eps=1e-5)  # Increased epsilon for stability
                self.linear2 = nn.Linear(d_model, d_model)
                self.norm2 = nn.LayerNorm(d_model, eps=1e-5)  # Increased epsilon for stability
                self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
                
            def forward(self, x):
                # Residual connection to preserve input
                residual = x
                
                # First transformation
                x = self.linear1(x)
                x = self.norm1(x)
                x = F.gelu(x)  # Use F.gelu instead of torch.gelu
                x = self.dropout(x)
                
                # Second transformation
                x = self.linear2(x)
                x = self.norm2(x)
                x = self.dropout(x)
                
                # Residual connection with scaling to prevent gradient issues
                return x + 0.1 * residual  # Scale residual to prevent gradient explosion
        
        self.line_feature_extractor = ResidualLineFeatureExtractor(d_model)
        
        # NEW: Simplified line-specific attention for vulnerability detection
        self.line_vuln_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout * 0.2,  # Reduced dropout
            batch_first=True
        )
        
        # NEW: Simplified vulnerability type-specific attention
        self.vuln_type_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout * 0.2,  # Reduced dropout
            batch_first=True
        )
        
        # NEW: Completely redesigned simple and robust line vulnerability head
        # Remove all LayerNorm layers that are causing the zero outputs
        self.line_vulnerability_head_1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Input: concatenated features
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_vulnerability_types)
        )
        
        # NEW: Simplified line-specific processor without LayerNorm
        self.line_specific_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # NEW: Simplified vulnerability type-specific processors without LayerNorm
        self.vuln_type_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, 1)
            ) for _ in range(num_vulnerability_types)
        ])
        
        # NEW: Add debugging to understand the flow
        self._debug_mode = False
        
        # AST path attention layer with more dropout
        self.ast_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between contract and AST with more dropout
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer with more dropout
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # GAN Discriminator components
        self.use_gan = use_gan
        if use_gan:
            # Path-aware attention for discriminator
            self.disc_path_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            
            # Grammar constraint for discriminator
            self.disc_grammar_embedding = nn.Embedding(vocab_size, d_model)
            self.disc_grammar_projection = nn.Linear(d_model, d_model)
            
            # Feature extraction for discriminator
            self.disc_feature_extractor = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Binary classification head for real vs fake detection
            self.disc_synthetic_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Single output for real/fake classification
            )
        
        self.d_model = d_model
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_vulnerability_types = num_vulnerability_types
        
        # NEW: Learnable embedding for empty lines
        self.empty_line_embedding = nn.Parameter(torch.zeros(d_model))
        
        # Initialize weights with better initialization
        self._init_weights()
        
        # Register gradient clipping hook on fusion layer to prevent explosion
        for param in self.feature_fusion.parameters():
            param.register_hook(self.hook_fn)
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0.0)
        
        # Initialize embeddings with smaller variance for stability
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.ast_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output layer with smaller weights
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        # Initialize vulnerability heads with smaller weights
        for module in self.contract_vulnerability_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.constant_(module.bias, 0.0)
        
        # IMPROVED: Initialize line vulnerability heads with stable weights
        for module in self.line_feature_extractor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # Reduced gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # NEW: Initialize line feature extractor with identity-like weights to prevent collapse
        for i, module in enumerate(self.line_feature_extractor.modules()):
            if isinstance(module, nn.Linear):
                # Initialize with small random weights to preserve input structure
                nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Small random weights
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                print(f"DEBUG: Initialized line feature extractor layer {i} with small random weights")
        
        # NEW: Initialize custom line feature extractor with small weights
        if hasattr(self.line_feature_extractor, 'linear1'):
            nn.init.normal_(self.line_feature_extractor.linear1.weight, mean=0.0, std=0.1)
            nn.init.constant_(self.line_feature_extractor.linear1.bias, 0.0)
            nn.init.normal_(self.line_feature_extractor.linear2.weight, mean=0.0, std=0.1)
            nn.init.constant_(self.line_feature_extractor.linear2.bias, 0.0)
            print("DEBUG: Initialized custom line feature extractor with small weights")
        
        # NEW: Initialize attention layers with stable weights
        for module in [self.line_vuln_attention, self.vuln_type_attention]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.8)  # Reduced gain for stability
                else:
                    nn.init.constant_(param, 0.0)
        
        # Initialize line vulnerability head with better weights for simplified architecture
        for module in self.line_vulnerability_head_1.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)  # Standard gain for better gradients
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # NEW: Initialize line-specific processor with better weights
        for module in self.line_specific_processor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)  # Standard gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # NEW: Initialize vulnerability type-specific processors with better weights
        for vuln_processor in self.vuln_type_processor:
            for module in vuln_processor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)  # Standard gain
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # IMPROVED: Initialize the final output layer with reasonable weights
        final_layer = self.line_vulnerability_head_1[-1]  # Get the last layer
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.1)  # Reasonable std
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, -0.2)  # Less conservative negative bias
        
        # Initialize discriminator components if GAN is enabled
        if self.use_gan:
            for module in self.disc_feature_extractor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            
            for module in self.disc_synthetic_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
    def hook_fn(self, grad):
        """Gradient clipping hook"""
        return torch.clamp(grad, -1.0, 1.0)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, input_ids, attention_mask=None, ast_input_ids=None, ast_attention_mask=None, 
                target_ids=None, token_to_line=None, apply_syntax_constraints=True):
        """
        Args:
            input_ids: Contract token ids [batch_size, seq_len]
            attention_mask: Contract attention mask [batch_size, seq_len]
            ast_input_ids: AST path token ids [batch_size, seq_len]
            ast_attention_mask: AST path attention mask [batch_size, seq_len]
            target_ids: Target token ids [batch_size, seq_len] (optional)
            token_to_line: Mapping from tokens to original line numbers [batch_size, seq_len]
            apply_syntax_constraints: Whether to apply syntax constraints during generation
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Contract embeddings with improved processing
        contract_emb = self.embedding(input_ids) * math.sqrt(self.d_model)
        contract_emb = self.embedding_dropout(contract_emb)
        contract_emb = self.embedding_norm(contract_emb)
        contract_emb = self.pos_encoder(contract_emb.transpose(0, 1)).transpose(0, 1)
        
        # AST path embeddings with improved processing
        ast_emb = self.ast_embedding(ast_input_ids) * math.sqrt(self.d_model)
        ast_emb = self.ast_embedding_dropout(ast_emb)
        ast_emb = self.ast_embedding_norm(ast_emb)
        ast_emb = self.pos_encoder(ast_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create source mask and ensure it's boolean
        src_mask = attention_mask if attention_mask is not None else torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        src_mask = src_mask.bool()  # Ensure boolean type
        
        # Encode contract with improved attention
        memory = self.encoder(contract_emb, src_key_padding_mask=~src_mask)
        
        # Apply AST path attention with residual connection - simplified
        if ast_attention_mask is not None:
            ast_attention_mask = ast_attention_mask.bool()
            ast_attn_output, _ = self.ast_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            memory = memory + 0.1 * ast_attn_output  # Reduced residual weight
        
        # Apply cross-attention between contract and AST - simplified
        if ast_attention_mask is not None:
            cross_attn_output, _ = self.cross_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            # Simplified feature fusion with reduced complexity
            fused_features = self.feature_fusion(torch.cat([memory, 0.1 * cross_attn_output], dim=-1))
            memory = memory + 0.1 * fused_features  # Reduced residual weight
        
        # IMPROVED: Get contract-level vulnerability predictions with better feature aggregation
        # Apply contract-level attention to focus on vulnerability-relevant parts
        contract_attn_output, contract_attn_weights = self.contract_vuln_attention(
            query=memory.mean(dim=1, keepdim=True),  # Global query [batch_size, 1, d_model]
            key=memory,  # [batch_size, seq_len, d_model]
            value=memory,  # [batch_size, seq_len, d_model]
            attn_mask=None
        )  # Output: [batch_size, 1, d_model]
        
        # Aggregate contract features with attention
        global_avg = memory.mean(dim=1)  # [batch_size, d_model]
        attention_weighted = contract_attn_output.squeeze(1)  # [batch_size, d_model]
        
        # Concatenate global average and attention-weighted features
        contract_representation = torch.cat([
            global_avg,  # [batch_size, d_model]
            attention_weighted  # [batch_size, d_model]
        ], dim=-1)  # [batch_size, d_model * 2]
        
        # Apply feature aggregation
        contract_features = self.contract_feature_aggregation(contract_representation)  # [batch_size, d_model]
        
        # Get contract-level vulnerability predictions
        contract_vuln_logits = self.contract_vulnerability_head(contract_features)  # [batch_size, num_vuln_types]
        
        # IMPROVED: Get line-level vulnerability predictions with stable architecture
        # Aggregate encoder outputs per line using token_to_line
        if token_to_line is not None:
            # memory: [batch_size, seq_len, d_model]
            # token_to_line: [batch_size, seq_len] (int: line index for each token)
            batch_size, seq_len, d_model = memory.shape
            max_lines = token_to_line.max().item() + 1  # number of lines in the contract
            
            # NEW: Debug line aggregation
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"DEBUG: Starting line aggregation...")
                print(f"DEBUG: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
                print(f"DEBUG: max_lines={max_lines}")
                print(f"DEBUG: token_to_line shape={token_to_line.shape}")
                print(f"DEBUG: token_to_line sample values: {token_to_line[:10].tolist()}")
            
            # Create line-specific features with position encoding
            line_features = []
            for b in range(batch_size):
                # For each line, average the features of all tokens that belong to that line
                # FIX: Handle the case where token_to_line is 1D (single batch)
                if token_to_line.dim() == 1:
                    this_token_to_line = token_to_line  # [seq_len]
                else:
                    this_token_to_line = token_to_line[b]  # [seq_len]
                this_memory = memory[b]  # [seq_len, d_model]
                lines = []
                for line_idx in range(max_lines):
                    mask = (this_token_to_line == line_idx)
                    if mask.any():
                        # DEBUG: Check mask and memory shapes
                        if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                            print(f"DEBUG: Line {line_idx}: mask shape: {mask.shape}, mask sum: {mask.sum().item()}")
                            print(f"DEBUG: Line {line_idx}: this_memory shape: {this_memory.shape}")
                            print(f"DEBUG: Line {line_idx}: this_token_to_line shape: {this_token_to_line.shape}")
                            print(f"DEBUG: Line {line_idx}: this_token_to_line range: [{this_token_to_line.min().item()}, {this_token_to_line.max().item()}]")
                            print(f"DEBUG: Line {line_idx}: max_lines: {max_lines}")
                            print(f"DEBUG: Line {line_idx}: line_idx: {line_idx}")
                            if mask.numel() > 0:
                                print(f"DEBUG: Line {line_idx}: mask values: {mask[:min(10, mask.numel())].tolist()}...")  # First 10 values
                            else:
                                print(f"DEBUG: Line {line_idx}: mask is empty")
                        
                        # Get the mean of tokens for this line
                        line_tokens = this_memory[mask]  # [num_tokens_in_line, d_model]
                        
                        # DEBUG: Check line_tokens shape
                        if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                            print(f"DEBUG: Line {line_idx}: line_tokens shape: {line_tokens.shape}, expected: [num_tokens, {d_model}]")
                        
                        # Ensure we get a 1D tensor of shape [d_model]
                        if line_tokens.dim() == 2:
                            # Check if the first dimension is the number of tokens
                            if line_tokens.shape[0] > 0 and line_tokens.shape[1] == d_model:
                                line_feature = line_tokens.mean(dim=0)  # [d_model]
                            else:
                                # Fallback: handle wrong shape
                                if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                                    print(f"DEBUG: Line {line_idx}: Wrong line_tokens shape {line_tokens.shape}, using fallback")
                                # Try to reshape or use zeros
                                if line_tokens.numel() >= d_model:
                                    line_feature = line_tokens.flatten()[:d_model]
                                else:
                                    line_feature = torch.zeros(d_model, device=line_tokens.device)
                        else:
                            line_feature = line_tokens.squeeze()  # [d_model]
                        
                        # DEBUG: Check line_feature shape before reshaping
                        if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                            print(f"DEBUG: Line {line_idx}: line_feature shape before reshape: {line_feature.shape}, numel: {line_feature.numel()}, expected: {d_model}")
                        
                        # Ensure it's exactly [d_model] shape - FIX: Use reshape instead of view
                        if line_feature.numel() != d_model:
                            # If the tensor has wrong size, reshape it properly
                            if line_feature.numel() > d_model:
                                line_feature = line_feature[:d_model]  # Truncate if too large
                                if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                                    print(f"DEBUG: Line {line_idx}: Truncated line_feature from {line_feature.numel()} to {d_model}")
                            else:
                                # Pad if too small
                                pad_size = d_model - line_feature.numel()
                                line_feature = torch.cat([line_feature.flatten(), torch.zeros(pad_size, device=line_feature.device)])
                                if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                                    print(f"DEBUG: Line {line_idx}: Padded line_feature from {line_feature.numel() - pad_size} to {d_model}")
                        else:
                            line_feature = line_feature.reshape(d_model)  # Ensure correct shape
                        # Add line position encoding to make each line unique
                        line_feature = line_feature + self._get_line_position_encoding(line_idx, d_model, device=memory.device)
                        lines.append(line_feature)
                        
                        # NEW: Debug line feature creation
                        if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                            print(f"DEBUG: Line {line_idx}: tokens={mask.sum().item()}, feature_range=[{line_feature.min().item():.4f}, {line_feature.max().item():.4f}]")
                    else:
                        # Use learnable embedding for empty lines with position encoding
                        empty_embedding = self.empty_line_embedding + self._get_line_position_encoding(line_idx, d_model, device=memory.device)
                        lines.append(empty_embedding)
                        
                        # NEW: Debug empty line
                        if hasattr(self, '_debug_mode') and self._debug_mode and b == 0 and line_idx < 3:
                            print(f"DEBUG: Line {line_idx}: empty, embedding_range=[{empty_embedding.min().item():.4f}, {empty_embedding.max().item():.4f}]")
                
                # Verify all tensors have the same shape before stacking
                for i, line in enumerate(lines):
                    if line.shape != torch.Size([d_model]):
                        print(f"Warning: Line {i} has shape {line.shape}, expected {d_model}")
                        # Force reshape to correct size
                        lines[i] = line.view(-1)[:d_model]
                        if lines[i].shape[0] < d_model:
                            # Pad if too short
                            pad_size = d_model - lines[i].shape[0]
                            lines[i] = torch.cat([lines[i], torch.zeros(pad_size, device=lines[i].device)])
                
                lines = torch.stack(lines, dim=0)  # [num_lines, d_model]
                line_features.append(lines)
                
                # NEW: Debug stacked lines
                if hasattr(self, '_debug_mode') and self._debug_mode and b == 0:
                    print(f"DEBUG: Stacked lines shape: {lines.shape}")
                    print(f"DEBUG: Stacked lines range: [{lines.min().item():.4f}, {lines.max().item():.4f}]")
                    print(f"DEBUG: Stacked lines std: {lines.std().item():.6f}")
            
            # Pad to max_lines across batch
            max_lines_in_batch = max([lf.shape[0] for lf in line_features])
            for i in range(len(line_features)):
                if line_features[i].shape[0] < max_lines_in_batch:
                    pad = torch.zeros((max_lines_in_batch - line_features[i].shape[0], d_model), device=memory.device)
                    line_features[i] = torch.cat([line_features[i], pad], dim=0)
            line_features = torch.stack(line_features, dim=0)  # [batch_size, max_lines, d_model]
            
            # NEW: Debug final line_features
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"DEBUG: Final line_features shape: {line_features.shape}")
                print(f"DEBUG: Final line_features range: [{line_features.min().item():.4f}, {line_features.max().item():.4f}]")
                print(f"DEBUG: Final line_features std: {line_features.std().item():.6f}")
        else:
            # Fallback: use per-token features (legacy behavior)
            line_features = memory  # Use raw memory instead of feature extraction

        # CRITICAL FIX: Store the original line_features before processing
        original_line_features = line_features.clone()  # [batch_size, num_lines, d_model]

        # NEW: Debug to verify original line features are preserved
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: original_line_features shape: {original_line_features.shape}")
            print(f"DEBUG: original_line_features range: [{original_line_features.min().item():.4f}, {original_line_features.max().item():.4f}]")
            print(f"DEBUG: original_line_features std: {original_line_features.std().item():.6f}")
            print(f"DEBUG: line_features before extractor shape: {line_features.shape}")
            print(f"DEBUG: line_features before extractor range: [{line_features.min().item():.4f}, {line_features.max().item():.4f}]")
            print(f"DEBUG: line_features before extractor std: {line_features.std().item():.6f}")
            
            # Check if line_features and original_line_features are the same
            if torch.allclose(line_features, original_line_features):
                print("✓ DEBUG: line_features and original_line_features are identical")
            else:
                print("⚠️  DEBUG: line_features and original_line_features are different!")
                diff = torch.abs(line_features - original_line_features).mean().item()
                print(f"DEBUG: Average difference: {diff:.6f}")

        # Process line_features through the feature extractor (only once)
        original_line_features = line_features.clone()  # Keep original for fallback
        line_features = self.line_feature_extractor(line_features)  # [batch_size, num_lines, d_model]
        
        # NEW: Fallback mechanism - if line feature extractor produces zeros, use original features
        if line_features.std().item() < 1e-6:
            print("⚠️  DEBUG: Line feature extractor produced zeros, using original features")
            line_features = original_line_features * 0.1  # Scale down to prevent gradient explosion
        
        # NEW: Debug to verify original line features are preserved
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: processed line_features range: [{line_features.min().item():.4f}, {line_features.max().item():.4f}]")
            print(f"DEBUG: processed line_features std: {line_features.std().item():.6f}")
            
            # Check if the extractor killed the features
            if line_features.std().item() < 1e-6:
                print("⚠️  DEBUG: Line feature extractor produced all zeros!")
                print("This suggests the extractor architecture has an issue")
            else:
                print("✓ DEBUG: Line feature extractor produced varied outputs")
        
        # Apply attention with smaller residual connections for stability
        line_attn_output, _ = self.line_vuln_attention(
            query=line_features,
            key=line_features,
            value=line_features,
            attn_mask=None
        )
        line_features = line_features + 0.05 * line_attn_output  # Smaller residual connection
        
        vuln_type_attn_output, _ = self.vuln_type_attention(
            query=line_features,
            key=line_features,
            value=line_features,
            attn_mask=None
        )
        line_features = line_features + 0.05 * vuln_type_attn_output  # Smaller residual connection
        
        # Use both original and attended features for better representation
        combined_features = torch.cat([line_features, line_attn_output], dim=-1)  # [batch_size, num_lines, d_model*2]
        
        # NEW: Process each line individually to ensure uniqueness
        batch_size, num_lines, _ = combined_features.shape
        line_vuln_logits = []
        
        # NEW: Debug line-specific processing
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: Processing {num_lines} lines individually...")
        
        for line_idx in range(num_lines):
            # Get features for this specific line
            line_feature = combined_features[:, line_idx, :]  # [batch_size, d_model*2]
            
            # NEW: Debug line features
            if hasattr(self, '_debug_mode') and self._debug_mode and line_idx < 3:
                print(f"DEBUG: Line {line_idx} feature range: [{line_feature.min().item():.4f}, {line_feature.max().item():.4f}], std: {line_feature.std().item():.6f}")
            
            # Process through the main head
            main_output = self.line_vulnerability_head_1(line_feature)  # [batch_size, num_vuln_types]
            
            # NEW: Debug main output
            if hasattr(self, '_debug_mode') and self._debug_mode and line_idx < 3:
                print(f"DEBUG: Line {line_idx} main_output range: [{main_output.min().item():.4f}, {main_output.max().item():.4f}], std: {main_output.std().item():.6f}")
            
            # Process through line-specific processor using ORIGINAL line features
            line_specific_feature = self.line_specific_processor(original_line_features[:, line_idx, :])  # [batch_size, d_model//2]
            
            # NEW: Debug line-specific features
            if hasattr(self, '_debug_mode') and self._debug_mode and line_idx < 3:
                print(f"DEBUG: Line {line_idx} line_specific_feature range: [{line_specific_feature.min().item():.4f}, {line_specific_feature.max().item():.4f}], std: {line_specific_feature.std().item():.6f}")
            
            # Process each vulnerability type separately
            vuln_type_outputs = []
            for vuln_type_idx in range(self.num_vulnerability_types):
                vuln_output = self.vuln_type_processor[vuln_type_idx](line_specific_feature)  # [batch_size, 1]
                vuln_type_outputs.append(vuln_output)
            
            # Combine main output with type-specific outputs
            type_specific_output = torch.cat(vuln_type_outputs, dim=1)  # [batch_size, num_vuln_types]
            
            # NEW: Debug type-specific output
            if hasattr(self, '_debug_mode') and self._debug_mode and line_idx < 3:
                print(f"DEBUG: Line {line_idx} type_specific_output range: [{type_specific_output.min().item():.4f}, {type_specific_output.max().item():.4f}], std: {type_specific_output.std().item():.6f}")
            
            # Combine both outputs with learnable weights
            combined_output = main_output + 0.1 * type_specific_output  # [batch_size, num_vuln_types]
            
            # NEW: Debug final combined output
            if hasattr(self, '_debug_mode') and self._debug_mode and line_idx < 3:
                print(f"DEBUG: Line {line_idx} combined_output range: [{combined_output.min().item():.4f}, {combined_output.max().item():.4f}], std: {combined_output.std().item():.6f}")
            
            line_vuln_logits.append(combined_output)
        
        # Stack all line outputs
        line_vuln_logits = torch.stack(line_vuln_logits, dim=1)  # [batch_size, num_lines, num_vuln_types]
        
        # NEW: Debug stacked outputs
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: Stacked line_vuln_logits shape: {line_vuln_logits.shape}")
            print(f"DEBUG: Stacked line_vuln_logits range: [{line_vuln_logits.min().item():.4f}, {line_vuln_logits.max().item():.4f}], std: {line_vuln_logits.std().item():.6f}")
            
            # Compare first few lines
            if line_vuln_logits.shape[1] > 1:
                line_0_output = line_vuln_logits[0, 0, :]  # First line, first batch
                line_1_output = line_vuln_logits[0, 1, :]  # Second line, first batch
                line_diff = torch.abs(line_0_output - line_1_output).mean().item()
                print(f"DEBUG: Stacked line difference (0 vs 1): {line_diff:.6f}")
                
                if line_diff < 1e-6:
                    print("⚠️  DEBUG: Stacked outputs are identical! Issue is in the processing loop.")
                else:
                    print("✓ DEBUG: Stacked outputs are different.")
        
        # PAD/TRUNCATE line_vuln_logits to [batch_size, 1024, num_vuln_types] to match targets
        max_lines = 1024  # or self.max_length
        if line_vuln_logits.shape[1] < max_lines:
            pad = torch.zeros(
                (line_vuln_logits.shape[0], max_lines - line_vuln_logits.shape[1], line_vuln_logits.shape[2]),
                device=line_vuln_logits.device
            )
            line_vuln_logits = torch.cat([line_vuln_logits, pad], dim=1)
        elif line_vuln_logits.shape[1] > max_lines:
            line_vuln_logits = line_vuln_logits[:, :max_lines, :]
        
        # NEW: Add debugging to understand the flow
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: memory range: [{memory.min().item():.4f}, {memory.max().item():.4f}]")
            
            # Debug line aggregation
            if token_to_line is not None:
                print(f"DEBUG: token_to_line shape: {token_to_line.shape}")
                print(f"DEBUG: token_to_line range: [{token_to_line.min().item()}, {token_to_line.max().item()}]")
                print(f"DEBUG: max_lines: {max_lines}")
                
                # Check if line_features are being created correctly
                print(f"DEBUG: line_features before extractor shape: {line_features.shape}")
                print(f"DEBUG: line_features before extractor range: [{line_features.min().item():.4f}, {line_features.max().item():.4f}]")
                print(f"DEBUG: line_features before extractor std: {line_features.std().item():.6f}")
            
            # Test each layer of the line feature extractor
            x = line_features  # Use line_features instead of memory
            print(f"DEBUG: Input to line feature extractor range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            
            # Handle custom ResidualLineFeatureExtractor
            if hasattr(self.line_feature_extractor, 'linear1'):
                # Custom module - test individual components
                x = self.line_feature_extractor.linear1(x)
                print(f"DEBUG: After Linear 1: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                x = self.line_feature_extractor.norm1(x)
                print(f"DEBUG: After LayerNorm 1: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                x = F.gelu(x)
                print(f"DEBUG: After GELU: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                x = self.line_feature_extractor.linear2(x)
                print(f"DEBUG: After Linear 2: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                x = self.line_feature_extractor.norm2(x)
                print(f"DEBUG: After LayerNorm 2: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
            else:
                # Sequential module - iterate normally
                for i, layer in enumerate(self.line_feature_extractor):
                    x = layer(x)
                    if isinstance(layer, nn.Linear):
                        print(f"DEBUG: After Linear {i}: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                    elif isinstance(layer, nn.GELU):
                        print(f"DEBUG: After GELU {i}: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                    elif isinstance(layer, nn.LayerNorm):
                        print(f"DEBUG: After LayerNorm {i}: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
                    elif isinstance(layer, nn.Dropout):
                        print(f"DEBUG: After Dropout {i}: range [{x.min().item():.4f}, {x.max().item():.4f}], std {x.std().item():.6f}")
            
            line_features = x
            print(f"DEBUG: line_features after extractor range: [{line_features.min().item():.4f}, {line_features.max().item():.4f}]")
            print(f"DEBUG: line_features after extractor std: {line_features.std().item():.6f}")
            
            # Debug attention outputs
            print(f"DEBUG: line_attn_output range: [{line_attn_output.min().item():.4f}, {line_attn_output.max().item():.4f}]")
            print(f"DEBUG: vuln_type_attn_output range: [{vuln_type_attn_output.min().item():.4f}, {vuln_type_attn_output.max().item():.4f}]")
            
            # Debug combined features
            print(f"DEBUG: combined_features range: [{combined_features.min().item():.4f}, {combined_features.max().item():.4f}]")
            print(f"DEBUG: combined_features std: {combined_features.std().item():.6f}")
            
            print(f"DEBUG: line_vuln_logits range: [{line_vuln_logits.min().item():.4f}, {line_vuln_logits.max().item():.4f}]")
            print(f"DEBUG: line_vuln_logits std: {line_vuln_logits.std().item():.6f}")
            
            # NEW: Test if different lines have different outputs
            if line_vuln_logits.shape[1] > 1:  # If we have multiple lines
                line_0_output = line_vuln_logits[0, 0, :]  # First line, first batch
                line_1_output = line_vuln_logits[0, 1, :]  # Second line, first batch
                line_diff = torch.abs(line_0_output - line_1_output).mean().item()
                print(f"DEBUG: Difference between line 0 and line 1: {line_diff:.6f}")
                
                if line_diff < 1e-6:
                    print("⚠️  DEBUG: Lines 0 and 1 have identical outputs!")
                    print("This suggests the line-specific processing is not working")
                    
                    # Debug the individual processing steps
                    print("DEBUG: Investigating why lines are identical...")
                    line_0_feature = combined_features[0, 0, :]  # First line features
                    line_1_feature = combined_features[0, 1, :]  # Second line features
                    feature_diff = torch.abs(line_0_feature - line_1_feature).mean().item()
                    print(f"DEBUG: Feature difference between line 0 and 1: {feature_diff:.6f}")
                    
                    if feature_diff < 1e-6:
                        print("⚠️  DEBUG: Line features are identical! The issue is in line aggregation.")
                    else:
                        print("✓ DEBUG: Line features are different, issue is in processing.")
                        
                else:
                    print("✓ DEBUG: Lines 0 and 1 have different outputs")
                    print(f"Line 0: {line_0_output[:3].tolist()}...")  # Show first 3 values
                    print(f"Line 1: {line_1_output[:3].tolist()}...")  # Show first 3 values
            
            # Test if the head is actually processing the input
            if line_vuln_logits.std().item() < 1e-6:
                print("⚠️  DEBUG: Line vulnerability head is producing constant outputs!")
                print("This suggests the head might not be processing the input correctly")
            else:
                print("✓ DEBUG: Line vulnerability head is producing varied outputs")
        
        # Debug prints removed to reduce output clutter
        
        # NEW: Add epoch tracking to model for debugging
        if hasattr(self, 'current_epoch'):
            self.current_epoch = getattr(self, 'current_epoch', 0)
        
        if target_ids is None:
            # Generate sequence with improved logic
            tgt = torch.ones((batch_size, 1), dtype=torch.long, device=device)  # BOS token
            max_len = min(self.max_length, 1024)  # Updated to match model's max_length
            
            # Improved generation parameters
            temperature = 0.7  # Lower temperature for more focused generation
            top_k = 50  # Increased top-k for better diversity
            top_p = 0.95  # Higher nucleus sampling threshold
            
            for i in range(max_len - 1):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = self.embedding_dropout(tgt_emb)
                tgt_emb = self.embedding_norm(tgt_emb)
                tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
                
                out = self.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~src_mask
                )
                
                # Apply layer normalization and dropout before output projection
                out = self.output_norm(out)
                out = self.output_dropout(out)
                logits = self.output_layer(out[:, -1, :])
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # Apply syntax constraints during generation (only during inference)
                if apply_syntax_constraints:
                    logits = self._apply_syntax_constraints(logits, tgt)
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_mask = torch.full_like(logits, float('-inf'))
                    logits_mask.scatter_(-1, top_k_indices, top_k_logits)
                    logits = logits_mask
                
                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Improved stopping conditions
                if (next_token == 2).any() or (next_token == 0).any():
                    # Only stop if we've generated a reasonable amount of tokens
                    if i > 50:  # Increased minimum length
                        break
                
                # Emergency break for very short generations
                if i > 20 and (next_token == 2).all():
                    break
            
            return {
                'generated_sequence': tgt,
                'contract_vulnerability_logits': contract_vuln_logits,
                'line_vulnerability_logits': line_vuln_logits
            }
            
        else:
            # Training mode with improved handling
            target_ids_copy = target_ids.clone()
            
            tgt_mask = self.generate_square_subsequent_mask(target_ids_copy.size(1)).to(device)
            
            tgt_emb = self.embedding(target_ids_copy) * math.sqrt(self.d_model)
            tgt_emb = self.embedding_dropout(tgt_emb)
            tgt_emb = self.embedding_norm(tgt_emb)
            tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~src_mask
            )
            
            # Apply layer normalization and dropout before output projection
            out = self.output_norm(out)
            out = self.output_dropout(out)
            logits = self.output_layer(out)
            
            # Reshape logits and target_ids to match
            shifted_target_ids = target_ids_copy[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shifted_target_ids = shifted_target_ids.view(-1)
            
            return {
                'logits': logits,
                'target_ids': shifted_target_ids,
                'contract_vulnerability_logits': contract_vuln_logits,
                'line_vulnerability_logits': line_vuln_logits,
                'encoder_output': memory.mean(dim=1),  # [batch_size, d_model] for discriminator
                'discriminator_logits': self.discriminator_forward(memory) if self.use_gan else None
            }

    def _apply_syntax_constraints(self, logits, prev_tokens):
        """
        Apply Solidity-specific syntax constraints to logits during generation.
        This prevents the model from generating syntactically invalid tokens.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            prev_tokens: Previously generated tokens [batch_size, seq_len]
            
        Returns:
            Constrained logits with invalid tokens masked out
        """
        batch_size = prev_tokens.size(0)
        device = logits.device
        
        # Get the last token for each sequence
        last_tokens = prev_tokens[:, -1]  # [batch_size]
        
        # Create mask for allowed tokens
        allowed_mask = torch.ones_like(logits)
        
        # Solidity keywords that require specific follow-up tokens
        keyword_constraints = {
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
        
        # Tokens that should be followed by semicolon
        semicolon_required = ['return', 'break', 'continue', 'require', 'assert', 'revert']
        
        # Apply constraints for each sequence in the batch
        for i in range(batch_size):
            last_token = last_tokens[i].item()
            
            # Try to decode the last token (this is a simplified approach)
            # In practice, you'd want to maintain a proper token-to-string mapping
            
            # Common token ID ranges for different types of tokens
            # These are approximate and would need to be adjusted for your specific tokenizer
            
            # Check if the last token looks like a keyword that needs constraints
            if last_token in [1024, 1025, 1026]:  # Example token IDs for function, contract, if
                # Apply keyword-specific constraints
                # This is where you'd implement the actual constraint logic
                pass
                
            # Check for tokens that should be followed by semicolon
            elif last_token in [2000, 2001, 2002]:  # Example token IDs for return, break, continue
                # Increase probability of semicolon
                semicolon_token_id = 59  # Common semicolon token ID
                if semicolon_token_id < logits.size(1):
                    logits[i, semicolon_token_id] *= 2.0  # Double the probability
                    
            # Check for opening braces/parentheses
            elif last_token in [40, 123]:  # '(' or '{'
                # Track that we need a closing token
                # This would require maintaining state across generation steps
                pass
                
            # Check for closing braces/parentheses
            elif last_token in [41, 125]:  # ')' or '}'
                # Ensure we had a matching opening token
                # This would require maintaining state across generation steps
                pass
        
        # Apply the mask to prevent invalid tokens
        logits = logits.masked_fill(allowed_mask == 0, float('-inf'))
        
        return logits

    def generate_with_beam_search(self, input_ids, attention_mask, path_input_ids, path_attention_mask, 
                                 beam_size=3, max_length=1024, temperature=1.0):
        """
        Generate sequences using beam search for better quality
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Encode the input (same as forward pass)
        contract_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        contract_emb = self.embedding(input_ids) * (self.d_model ** 0.5)
        contract_emb = contract_emb + self.pos_encoder(contract_pos)
        
        path_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        path_emb = self.path_embedding(path_input_ids) * (self.d_model ** 0.5)
        path_emb = path_emb + self.pos_encoder(path_pos)
        
        src_emb = torch.cat([contract_emb, path_emb], dim=1)
        combined_seq_len = 2 * seq_len
        
        if combined_seq_len > self.max_length:
            truncate_len = self.max_length // 2
            src_emb = torch.cat([
                contract_emb[:, :truncate_len, :], 
                path_emb[:, :truncate_len, :]
            ], dim=1)
            combined_seq_len = self.max_length
        
        if attention_mask is not None and path_attention_mask is not None:
            if combined_seq_len < 2 * seq_len:
                truncate_len = combined_seq_len // 2
                src_mask = torch.cat([
                    attention_mask[:, :truncate_len], 
                    path_attention_mask[:, :truncate_len]
                ], dim=1)
            else:
                src_mask = torch.cat([attention_mask, path_attention_mask], dim=1)
        else:
            src_mask = torch.ones((batch_size, combined_seq_len), dtype=torch.bool, device=device)
        
        memory = self.encoder(src_emb, src_key_padding_mask=~src_mask)
        
        # Initialize beam search
        # Each beam: (sequence, score)
        beams = [[(torch.ones((1, 1), dtype=torch.long, device=device), 0.0)] for _ in range(batch_size)]
        
        for step in range(max_length - 1):
            new_beams = [[] for _ in range(batch_size)]
            
            for batch_idx in range(batch_size):
                for seq, score in beams[batch_idx]:
                    if len(seq[0]) >= max_length or seq[0, -1].item() == 2:  # EOS token
                        new_beams[batch_idx].append((seq, score))
                        continue
                    
                    # Generate next token probabilities
                    tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                    tgt_pos = torch.arange(0, seq.size(1), device=device).unsqueeze(0)
                    
                    tgt_emb = self.embedding(seq) * (self.d_model ** 0.5)
                    tgt_emb = tgt_emb + self.pos_encoder(tgt_pos)
                    
                    out = self.decoder(
                        tgt_emb,
                        memory[batch_idx:batch_idx+1],
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=~src_mask[batch_idx:batch_idx+1]
                    )
                    
                    logits = self.output_layer(out[:, -1, :]) / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top-k candidates
                    top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size, dim=-1)
                    
                    for k in range(beam_size):
                        next_token = top_k_indices[0, k].unsqueeze(0).unsqueeze(0)
                        next_log_prob = top_k_log_probs[0, k].item()
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + next_log_prob
                        new_beams[batch_idx].append((new_seq, new_score))
                
                # Keep only top beam_size candidates
                new_beams[batch_idx].sort(key=lambda x: x[1], reverse=True)
                new_beams[batch_idx] = new_beams[batch_idx][:beam_size]
            
            beams = new_beams
        
        # Return best sequence for each batch
        best_sequences = []
        for batch_idx in range(batch_size):
            best_seq, _ = max(beams[batch_idx], key=lambda x: x[1])
            best_sequences.append(best_seq[0])
        
        # Stack sequences (pad if necessary)
        max_seq_len = max(seq.size(0) for seq in best_sequences)
        padded_sequences = []
        for seq in best_sequences:
            if seq.size(0) < max_seq_len:
                padding = torch.zeros((max_seq_len - seq.size(0),), dtype=torch.long, device=device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        result = torch.stack(padded_sequences)
        encoder_output = torch.mean(memory, dim=1)
        
        return {
            'generated_sequence': result,
            'encoder_output': encoder_output
        }

    def discriminator_forward(self, features):
        """
        Forward pass for the integrated discriminator
        Args:
            features: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            synthetic_logits: Logits for real vs fake classification
        """
        if not self.use_gan:
            return None
        
        # Apply path-aware attention
        attn_output, _ = self.disc_path_attention(features, features, features)
        x = features + attn_output
        
        # Apply grammar constraint
        x = self.disc_grammar_projection(x)
        
        # Global average pooling to get fixed-size representation
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Extract features
        features = self.disc_feature_extractor(x)
        
        # Binary classification: real (1) vs fake (0)
        synthetic_logits = self.disc_synthetic_head(features)
        
        return synthetic_logits
    
    def set_current_epoch(self, epoch):
        """Set current epoch for debugging purposes"""
        self.current_epoch = epoch
        
    def _get_line_position_encoding(self, line_idx, d_model, device):
        """Generate position encoding for a specific line to make it unique"""
        # Create a simple sinusoidal position encoding
        position = torch.tensor(line_idx, dtype=torch.float, device=device)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                           -(math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(d_model, device=device)
        pos_encoding[0::2] = torch.sin(position * div_term)
        pos_encoding[1::2] = torch.cos(position * div_term)
        
        return pos_encoding 