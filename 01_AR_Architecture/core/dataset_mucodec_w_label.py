import os
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class MusicTextDataset_delay(Dataset):
    """
    Dataset for Autoregressive Music Generation.
    Implements the 'delay pattern' for multi-codebook sequence modeling and 
    supports text dropout for Classifier-Free Guidance (CFG).
    """
    def __init__(
        self,
        token_dir,
        block_size=1024,
        stride=256,
        rate=1.0,       
        seed=42,        
        bos_token=16384,     
        codec_name="mucodec",
        drop_text_prob=0.1   # Probability to drop text condition for CFG
    ):
        self.block_size = block_size
        self.stride = stride
        self.bos_token = bos_token
        self.drop_text_prob = drop_text_prob 
        self.sample_index = []  

        cache_file = os.path.join(token_dir, f"index_cache_{codec_name}_b{block_size}_s{stride}.pt")
        
        # Load or build dataset index
        if os.path.exists(cache_file):
            print(f"Loading cached dataset index from {cache_file}...")
            self.sample_index = torch.load(cache_file)
        else:
            print("Building dataset index from scratch (this may take a few minutes)...")
            files = sorted([
                os.path.join(token_dir, f)
                for f in os.listdir(token_dir)
                if f.endswith(".pt") and not f.startswith("index_cache")
            ])

            for f in tqdm(files, desc="Scanning files"):
                data = torch.load(f, map_location="cpu", weights_only=False)
                # Calculate total tokens across all chunks
                T = sum([chunk.shape[1] for chunk in data["tokens"]])

                # Create sliding window indices
                for i in range(0, T - block_size - 1, stride):
                    self.sample_index.append((f, i))
            
            torch.save(self.sample_index, cache_file)
            print(f"Full index built and cached! Total samples: {len(self.sample_index)}")

        # Subsample dataset if rate < 1.0 (useful for debugging/testing)
        if 0.0 < rate < 1.0:
            total_samples = len(self.sample_index)
            keep_num = int(total_samples * rate)
            
            random.seed(seed)
            random.shuffle(self.sample_index)
            self.sample_index = self.sample_index[:keep_num]
            
            print(f"📉 Subsampled dataset at rate {rate*100:.1f}%: Using {keep_num} / {total_samples} samples.")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_path, start_idx = self.sample_index[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # --- 1. Process Audio Tokens ---
        chunks = data["tokens"]            
        tokens = torch.cat(chunks, dim=1)  
        tokens = tokens.permute(1, 0)      
        
        K = tokens.shape[1] # Number of codebooks (e.g., 4 or 8)
        req_len = self.block_size + 1 + K
        
        # Pad sequence if it's too short
        if start_idx + req_len > tokens.shape[0]:
            pad_len = (start_idx + req_len) - tokens.shape[0]
            pad_tensor = torch.full((pad_len, K), self.bos_token, dtype=tokens.dtype) 
            raw_seq = torch.cat([tokens[start_idx:], pad_tensor], dim=0)
        else:
            raw_seq = tokens[start_idx : start_idx + req_len]
            
        # Apply Delay Pattern for multi-codebook AR modeling
        delayed_seq = torch.full((self.block_size + 1, K), self.bos_token, dtype=tokens.dtype)
        
        for k in range(K):
            valid_len = self.block_size + 1 - k
            delayed_seq[k : k + valid_len, k] = raw_seq[0 : valid_len, k]

        x = delayed_seq[:-1, :]  # Input sequence
        y = delayed_seq[1:, :]   # Target sequence (shifted by 1)
        
        # --- 2. Process Text Condition (with CFG Dropout) ---
        is_drop = random.random() < self.drop_text_prob
        
        if is_drop:
            # Unconditional generation: return empty prompt and zeroed embeddings
            text_embeds = torch.zeros((1, 768), dtype=torch.float32)
            text_prompt = ""  
        else:
            # Conditional generation: extract actual text embeddings
            text_embeds = data.get("text_embeds", torch.zeros((1, 1, 768)))
            text_embeds = text_embeds.squeeze(0) 
            text_prompt = data.get("text_prompt", "No prompt found.")
        
        return x, y, text_embeds, text_prompt


# ==========================================
# Collate Function for DataLoader
# Handles dynamic padding for variable-length text embeddings
# ==========================================
def collate_fn_with_text(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    embeds = [item[2] for item in batch]
    prompts = [item[3] for item in batch]

    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)

    # Pad text embeddings to the maximum length in the current batch
    text_lengths = torch.tensor([emb.shape[0] for emb in embeds])
    embeds_padded = pad_sequence(embeds, batch_first=True, padding_value=0.0)

    # Generate boolean attention mask for the text embeddings
    max_len = embeds_padded.shape[1]
    attention_mask = torch.arange(max_len).expand(len(text_lengths), max_len) < text_lengths.unsqueeze(1)

    return x_batch, y_batch, embeds_padded, attention_mask, prompts


if __name__ == "__main__":
    # ==========================================
    # Quick Test Block
    # ==========================================
    # Update this to your local parsed dataset directory
    data_dir = "./data/MTG_dataset/mtg_tokens_mucodec_val_w_MSMlabel" 
    
    train_dataset = MusicTextDataset_delay(
        token_dir=data_dir,
        block_size=1024, 
        stride=256,      
        rate=1.0,
        bos_token=16384, 
        codec_name="mucodec",
        drop_text_prob=0.1
    )
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn_with_text
    )
    
    # Fetch a single batch to verify shapes
    for batch_idx, (x, y, text_embeds, attention_mask, prompts) in enumerate(dataloader):
        print("\n=== Batch Shape Verification ===")
        print(f"x shape: {x.shape}  <-- Expected: [Batch, Block_Size, Codebooks]")                           
        print(f"y shape: {y.shape}  <-- Expected: [Batch, Block_Size, Codebooks]")                           
        print(f"text_embeds shape: {text_embeds.shape} <-- Expected: [Batch, Max_Text_Len, 768]")       
        print(f"attention_mask shape: {attention_mask.shape} <-- Expected: [Batch, Max_Text_Len]") 
        
        print("\n=== Text Prompts Check ===")
        for i, prompt in enumerate(prompts):
            # Empty strings indicate the CFG dropout was triggered
            status = " [DROPPED FOR CFG]" if prompt == "" else ""
            print(f"Sample {i+1}: '{prompt}'{status}")
            
        break
