import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from transformers import AutoModel, AutoTokenizer

class SmartContractTransformer(nn.Module):
    def __init__(
        self,
        d_model=768,  # Hidden size for embeddings and transformer layers
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=512,  # Match your dataset's max_length
        vocab_size=50265  # Match your tokenizer's vocab size
    ):
        super().__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_length, d_model)
        
        # Path embedding (separate embedding for path tokens)
        self.path_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # Initialize biases to small values
                nn.init.constant_(p, 0.0)
        
        # Initialize embeddings with smaller variance for stability
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.path_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_encoder.weight, mean=0.0, std=0.02)
                
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, input_ids, attention_mask=None, path_input_ids=None, path_attention_mask=None, target_ids=None):
        """
        Args:
            input_ids: Contract token ids [batch_size, seq_len]
            attention_mask: Contract attention mask [batch_size, seq_len]
            path_input_ids: Path token ids [batch_size, seq_len]
            path_attention_mask: Path attention mask [batch_size, seq_len]
            target_ids: Target token ids [batch_size, seq_len]
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Create position indices for contract tokens
        contract_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Contract embeddings
        contract_emb = self.embedding(input_ids) * (self.d_model ** 0.5)
        contract_emb = contract_emb + self.pos_encoder(contract_pos)
        
        # Path embeddings with proper position indices
        path_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        path_emb = self.path_embedding(path_input_ids) * (self.d_model ** 0.5)
        path_emb = path_emb + self.pos_encoder(path_pos)
        
        # Combine contract and path embeddings
        src_emb = torch.cat([contract_emb, path_emb], dim=1)
        
        # Create position indices for the concatenated sequence
        combined_seq_len = 2 * seq_len
        if combined_seq_len > self.max_length:
            # Truncate if too long
            truncate_len = self.max_length // 2
            src_emb = torch.cat([
                contract_emb[:, :truncate_len, :], 
                path_emb[:, :truncate_len, :]
            ], dim=1)
            combined_seq_len = self.max_length
        
        # Create source mask - ensure it's 2D
        if attention_mask is not None and path_attention_mask is not None:
            if combined_seq_len < 2 * seq_len:
                # Truncate masks if we truncated embeddings
                truncate_len = combined_seq_len // 2
                src_mask = torch.cat([
                    attention_mask[:, :truncate_len], 
                    path_attention_mask[:, :truncate_len]
                ], dim=1)
            else:
                src_mask = torch.cat([attention_mask, path_attention_mask], dim=1)
        else:
            src_mask = torch.ones((batch_size, combined_seq_len), dtype=torch.bool, device=device)
        
        # Encode - use the 2D mask directly
        memory = self.encoder(src_emb, src_key_padding_mask=~src_mask)
        
        if target_ids is None:
            # Generate sequence
            tgt = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            max_len = min(self.max_length, 50)  # Limit generation length to prevent infinite loops
            
            for _ in range(max_len - 1):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                tgt_pos = torch.arange(0, tgt.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
                
                tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
                tgt_emb = tgt_emb + self.pos_encoder(tgt_pos)
                
                out = self.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~src_mask
                )
                
                out = self.output_layer(out[:, -1:])
                next_token = torch.argmax(out, dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if we predict the end token
                if (next_token == 2).all():  # Assuming 2 is the end token
                    break
            
            # Get the mean of the encoder output for the discriminator
            encoder_output = torch.mean(memory, dim=1)  # [batch_size, d_model]
                    
            return {
                'generated_sequence': tgt,
                'encoder_output': encoder_output
            }
            
        else:
            # Training mode - use a copy of target_ids to avoid modifying the original
            target_ids_copy = target_ids.clone()
            
            tgt_pos = torch.arange(0, target_ids_copy.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_mask = self.generate_square_subsequent_mask(target_ids_copy.size(1)).to(device)
            
            tgt_emb = self.embedding(target_ids_copy) * (self.d_model ** 0.5)
            tgt_emb = tgt_emb + self.pos_encoder(tgt_pos)
            
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~src_mask
            )
            
            logits = self.output_layer(out)
            
            # Reshape logits and target_ids to match
            # Remove the first token from target_ids (shift right)
            shifted_target_ids = target_ids_copy[:, 1:].contiguous()
            
            # Reshape both to [batch_size * (seq_len-1), vocab_size]
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shifted_target_ids = shifted_target_ids.view(-1)
            
            # Get the mean of the encoder output for the discriminator
            encoder_output = torch.mean(memory, dim=1)  # [batch_size, d_model]
            
            return {
                'logits': logits,
                'target_ids': shifted_target_ids,
                'encoder_output': encoder_output
            } 