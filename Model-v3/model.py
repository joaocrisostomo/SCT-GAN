import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from transformers import AutoModel, AutoTokenizer
import math

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
        
        # Contract-level vulnerability detection head with more dropout
        self.contract_vulnerability_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_vulnerability_types)
        )
        
        # IMPROVED: Line-level vulnerability detection with spatial attention
        # Spatial attention for capturing local context
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Line-level vulnerability detection head with spatial context
        self.line_vulnerability_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for spatial context
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_vulnerability_types)
        )
        
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
        
        for module in self.line_vulnerability_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.constant_(module.bias, 0.0)
        
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
                target_ids=None, token_to_line=None):
        """
        Args:
            input_ids: Contract token ids [batch_size, seq_len]
            attention_mask: Contract attention mask [batch_size, seq_len]
            ast_input_ids: AST path token ids [batch_size, seq_len]
            ast_attention_mask: AST path attention mask [batch_size, seq_len]
            target_ids: Target token ids [batch_size, seq_len] (optional)
            token_to_line: Mapping from tokens to original line numbers [batch_size, seq_len]
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
        
        # Get contract-level vulnerability predictions
        contract_representation = torch.mean(memory, dim=1)  # [batch_size, d_model]
        contract_vuln_logits = self.contract_vulnerability_head(contract_representation)  # [batch_size, num_vuln_types]
        
        # IMPROVED: Get line-level vulnerability predictions with spatial context
        # Apply spatial attention to capture local context
        spatial_context, _ = self.spatial_attention(
            query=memory,
            key=memory,
            value=memory,
            attn_mask=None  # Allow full attention for spatial context
        )
        
        # Combine original features with spatial context
        line_features = torch.cat([memory, spatial_context], dim=-1)  # [batch_size, seq_len, d_model*2]
        line_vuln_logits = self.line_vulnerability_head(line_features)  # [batch_size, seq_len, num_vuln_types]
        
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