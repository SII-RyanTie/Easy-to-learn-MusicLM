import torch
import torch.nn as nn

class MusicTransformerText_delay(nn.Module):
    """
    Autoregressive Transformer Decoder for Text-to-Music generation.
    
    This model uses a Transformer Decoder architecture where:
    - Target Sequence: Autoregressive audio tokens (with causal mask).
    - Memory Sequence: Text embeddings injected via Cross-Attention.
    """
    def __init__(
        self, 
        vocab_size=16384,        # Default vocabulary size for MuCodec
        num_codebooks=1,         # MuCodec uses a single codebook
        embed_dim=1280, 
        text_embed_dim=768,      # Dimension of FLAN-T5 text embeddings
        max_seq_len=4096,
        num_layers=24, 
        num_heads=16, 
        dropout=0.1
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim

        # Note: Vocabulary size is +1 to accommodate the BOS/PAD token (e.g., 16384)
        self.vocab_size_with_bos = vocab_size + 1

        # 1. Independent Embeddings for each Codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size_with_bos, embed_dim)
            for _ in range(num_codebooks)
        ])

        # 2. Temporal Positional Encoding
        self.time_pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 3. Text Feature Projection Layer
        # Projects text embeddings (e.g., 768d) to the model's hidden dimension
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        # 4. Transformer Decoder with Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True  
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Output Prediction Heads (Predicting actual vocab_size without BOS)
        self.output_heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size)
            for _ in range(num_codebooks)
        ])

    def forward(self, x, text_embeds, text_mask=None):
        """
        Args:
            x: Audio tokens of shape [Batch, Time, Codebooks]
            text_embeds: Text embeddings of shape [Batch, Seq_Len, Text_Dim]
            text_mask: Boolean mask of shape [Batch, Seq_Len]. 
                       True indicates valid tokens, False indicates padding.
        Returns:
            logits: Prediction logits of shape [Batch, Time, Codebooks, Vocab_Size]
        """
        B, T, K = x.shape
        
        # --- A. Process Audio Features (Target) ---
        h = 0
        for k in range(K):
            h = h + self.embeddings[k](x[:, :, k]) 

        time_ids = torch.arange(T, device=x.device)
        time_emb = self.time_pos_embedding(time_ids) 
        h = h + time_emb.unsqueeze(0)                

        # --- B. Process Text Features (Memory) ---
        # Project to match audio embedding dimension: [B, Seq_Len, embed_dim]
        memory = self.text_proj(text_embeds)

        # --- C. Generate Attention Masks ---
        # 1. Causal mask for autoregressive audio generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # 2. Padding mask for text embeddings
        # Note: PyTorch Transformer expects True for positions that SHOULD be masked (ignored)
        if text_mask is not None:
            memory_key_padding_mask = ~text_mask
        else:
            memory_key_padding_mask = None

        # --- D. Transformer Decoder Forward Pass ---
        h = self.transformer(
            tgt=h, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True 
        ) 

        # --- E. Independent Codebook Predictions ---
        logits_list = []
        for k in range(K):
            logits_k = self.output_heads[k](h) 
            logits_list.append(logits_k)

        # Stack logits along the codebook dimension
        logits = torch.stack(logits_list, dim=2)
        
        return logits

# ==========================================
# Quick Verification Block
# ==========================================
if __name__ == "__main__":
    # Simulate a batch of data from the Dataset (MuCodec single codebook)
    B, T, K = 2, 512, 1
    seq_len = 15 
    
    dummy_x = torch.randint(0, 16385, (B, T, K))
    dummy_text_embeds = torch.randn(B, seq_len, 768)
    
    # Simulate Attention Mask (False means padded token)
    dummy_text_mask = torch.ones(B, seq_len, dtype=torch.bool)
    dummy_text_mask[1, 10:] = False 

    print("=== Input Shapes ===")
    print(f"Audio x:           {dummy_x.shape}")
    print(f"Text embeds:       {dummy_text_embeds.shape}")
    print(f"Text mask:         {dummy_text_mask.shape}")

    # Initialize the model (0.5B scale configuration)
    model = MusicTransformerText_delay(
        vocab_size=16384,
        num_codebooks=1,
        embed_dim=1280,      
        num_layers=24,       
        num_heads=16,        
        dropout=0.1          
    )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== Model Statistics ===")
    print(f"Total Parameters:     {total_params / 1e9:.4f} B")
    print(f"Trainable Parameters: {trainable_params / 1e9:.4f} B")

    # Forward pass
    logits = model(dummy_x, dummy_text_embeds, dummy_text_mask)

    print("\n=== Output Shape ===")
    print(f"Logits shape: {logits.shape}  <-- Expected: [{B}, {T}, {K}, 16384]")
