import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
import numpy as np
        
class SmartContractTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        # CodeBERT outputs [batch_size, seq_length, 768]
        # Apply transformer
        x = self.transformer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x

class Generator(nn.Module):
    def __init__(self, d_model=768, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.main = nn.Sequential(
            # Input layer
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(dim_feedforward // 2, d_model),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        # Ensure input is 2D [batch_size, features]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, d_model=768, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.main = nn.Sequential(
            # Input layer
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            # Hidden layer
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        # Ensure input is 2D [batch_size, features]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # Add safety check for input dimensions
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {x.size(-1)}")
            
        return self.main(x)

class CodeDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=50000, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Increase model capacity
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Improve embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Improve output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Special tokens for Solidity syntax
        self.special_tokens = {
            'pragma': 0,
            'solidity': 1,
            'contract': 2,
            'function': 3,
            'returns': 4,
            'public': 5,
            'private': 6,
            'view': 7
        }
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, memory, target_sequence=None):
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate target sequence if not provided
        if target_sequence is None:
            batch_size = x.size(0)
            # Start with pragma token
            target_sequence = torch.full((batch_size, 1), self.special_tokens['pragma'], device=x.device)
            
            # Generate sequence
            for _ in range(512):  # Maximum sequence length
                # Get embeddings
                target_embeddings = self.embedding(target_sequence)
                target_embeddings = self.pos_encoder(target_embeddings)
                
                # Apply transformer
                output = self.transformer(
                    target_embeddings,
                    memory,
                    tgt_mask=self._generate_square_subsequent_mask(target_sequence.size(1)).to(x.device)
                )
                
                # Project to vocabulary
                logits = self.output_projection(output[:, -1:])
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)
                
                # Append to sequence
                target_sequence = torch.cat([target_sequence, next_token], dim=1)
                
                # Stop if we generate an end token
                if (next_token == 0).all():
                    break
        
        # Get embeddings for target sequence
        target_embeddings = self.embedding(target_sequence)
        target_embeddings = self.pos_encoder(target_embeddings)
        
        # Apply transformer
        output = self.transformer(
            target_embeddings,
            memory,
            tgt_mask=self._generate_square_subsequent_mask(target_sequence.size(1)).to(x.device)
        )
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SmartContractVulnerabilityGAN(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=50000, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize CodeBERT
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.codebert.eval()  # Keep CodeBERT in eval mode
        for param in self.codebert.parameters():
            param.requires_grad = False
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Initialize transformer
        self.transformer = SmartContractTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Initialize generator
        self.generator = Generator(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Initialize decoder
        self.decoder = CodeDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size,
            max_length=max_length
        )
        
    def forward(self, contract_ids, path_ids, contract_attention_mask=None, path_attention_mask=None):
        # Get CodeBERT embeddings with no_grad
        with torch.no_grad():
            contract_outputs = self.codebert(
                input_ids=contract_ids,
                attention_mask=contract_attention_mask
            )
            contract_embeddings = contract_outputs.last_hidden_state
            
            path_outputs = self.codebert(
                input_ids=path_ids,
                attention_mask=path_attention_mask
            )
            path_embeddings = path_outputs.last_hidden_state
        
        # Combine embeddings
        combined_embeddings = contract_embeddings + path_embeddings  # [batch_size, seq_length, 768]
        
        # Process through transformer
        transformed = self.transformer(combined_embeddings)  # [batch_size, seq_length, 768]
        
        # Get mean representation
        mean_embeddings = transformed.mean(dim=1)  # [batch_size, 768]
        
        # Generate synthetic vulnerabilities
        synthetic = self.generator(mean_embeddings)  # [batch_size, 768]
        
        # Get discriminator scores
        real_scores = self.discriminator(mean_embeddings)
        fake_scores = self.discriminator(synthetic)
        
        # Decode synthetic embeddings back to code
        decoded_code = self.decoder(synthetic, transformed)  # [batch_size, seq_length, vocab_size]
        
        return {
            'embeddings': transformed,
            'synthetic': synthetic,
            'real_scores': real_scores,
            'fake_scores': fake_scores,
            'decoded_code': decoded_code
        }
    
    def decode_embeddings(self, embeddings):
        """
        Decode embeddings back to code
        """
        with torch.no_grad():
            return self.decoder(embeddings, embeddings)
    
    def generate_code(self, num_samples=1, beam_size=5):
        """
        Generate synthetic code using beam search
        """
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, 768).to(next(self.parameters()).device)
            
            # Generate synthetic embeddings
            synthetic_embeddings = self.generator(noise)
            
            # Create memory for decoder (use synthetic embeddings as memory)
            memory = synthetic_embeddings.unsqueeze(1)  # Add sequence dimension
            
            # Decode embeddings to code
            decoded_code = self.decoder(synthetic_embeddings, memory)
            
            # Convert to tokens
            tokens = torch.argmax(decoded_code, dim=-1)
            
            # Convert tokens to code
            code = self.tokenizer.decode(tokens[0])
            
            return code