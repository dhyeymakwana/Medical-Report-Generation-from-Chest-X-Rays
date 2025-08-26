# src/model.py (changes highlighted)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertTokenizer
from src.config import TOKENIZER_MODEL
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# --- NEW: Relational Memory Module ---
class MemoryModule(nn.Module):
    """
    A relational memory module inspired by R2Gen.
    It functions as an external memory to improve coherence.
    """
    def __init__(self, embed_dim, nhead=8):
        super().__init__()
        self.embed_dim = embed_dim
        # The memory is a learnable tensor
        self.memory = nn.Parameter(torch.randn(512, embed_dim)) # Memory size of 512
        
        # Multi-head attention for reading from and writing to memory
        self.read_attention = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.write_attention = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

    def forward(self, query):
        """
        Reads from and writes to the memory based on a query vector.
        
        Args:
            query (Tensor): The current hidden state of the decoder.
        """
        batch_size = query.size(0)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Read from memory
        read_output, _ = self.read_attention(query, memory_expanded, memory_expanded)
        
        # Write to memory (this is a simplified update mechanism)
        # In a full implementation, this would involve a more complex gating mechanism.
        # For simplicity, we use the query to update the memory via attention.
        write_output, _ = self.write_attention(query, memory_expanded, memory_expanded)
        
        # Update memory (simplified: a weighted sum of old and new)
        # Note: This update doesn't persist across batches in this simplified form.
        # A true stateful implementation would be more complex.
        updated_memory_info = write_output 
        
        return read_output + updated_memory_info


# --- UPDATED: ReportDecoder ---
class ReportDecoder(nn.Module):
    """
    Transformer-based report decoder, now enhanced with a MemoryModule.
    """
    # CHANGE THE __init__ SIGNATURE
    def __init__(self, embed_dim, num_layers, nhead, tokenizer, dropout=0.1):
        super().__init__()
        # STORE THE TOKENIZER AND GET VOCAB SIZE FROM IT
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        # ... (the rest of the __init__ method is the same)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.memory_module = MemoryModule(embed_dim, nhead)
        self.output_layer = nn.Linear(embed_dim * 2, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, tgt_tokens, tgt_mask=None, tgt_padding_mask=None):
        tgt_embed = self.dropout(self.embedding(tgt_tokens) + self.positional_encoding[:, :tgt_tokens.size(1)])
        
        # 1. Pass through the standard Transformer Decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 2. Use the decoder's output as a query for the memory module
        memory_output = self.memory_module(decoder_output)
        
        # 3. Concatenate the outputs from the decoder and the memory module
        combined_output = torch.cat([decoder_output, memory_output], dim=-1)
        
        # 4. Map to vocabulary to get final word predictions
        return self.output_layer(combined_output)

# --- UNCHANGED CLASSES ---
# The ImageEncoder and EncoderDecoderModel classes do not need any changes.
# The EncoderDecoderModel will automatically use the updated ReportDecoder.

class ImageEncoder(nn.Module):
    # Change 'vit_tiny...' back to 'vit_base...'
    def __init__(self, model_name='vit_base_patch16_224_in21k', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        original_conv = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        self.model.patch_embed.proj.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
    def forward(self, x):
        return self.model.forward_features(x)

class EncoderDecoderModel(nn.Module):
    def __init__(self, embed_dim=768, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL)
        self.pad_token_id = tokenizer.pad_token_id
        
        self.encoder = ImageEncoder(model_name='vit_base_patch16_224_in21k')
        # This line now correctly passes the entire tokenizer object
        self.decoder = ReportDecoder(embed_dim, num_layers, nhead, tokenizer, dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, image, tgt_tokens):
        cls_token = self.encoder.model.cls_token.expand(image.size(0), -1, -1)
        encoder_output = self.encoder(image)
        memory = torch.cat([cls_token, encoder_output], dim=1)
        
        tgt_padding_mask = (tgt_tokens == self.pad_token_id)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(image.device)
        
        logits = self.decoder(memory, tgt_tokens, tgt_mask, tgt_padding_mask)
        
        return logits
    
    def generate(self, image, max_len=256):
        self.eval()
        start_token_id = self.decoder.tokenizer.cls_token_id
        end_token_id = self.decoder.tokenizer.sep_token_id
        with torch.no_grad():
            cls_token = self.encoder.model.cls_token.expand(image.size(0), -1, -1)
            encoder_output = self.encoder(image)
            memory = torch.cat([cls_token, encoder_output], dim=1)
            tgt_tokens = torch.full((image.size(0), 1), start_token_id, dtype=torch.long).to(image.device)
            for _ in range(max_len - 1):
                tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(image.device)
                logits = self.decoder(memory, tgt_tokens, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                if (next_token == end_token_id).all():
                    break
        return tgt_tokens

    def forward(self, image, tgt_tokens):
        cls_token = self.encoder.model.cls_token.expand(image.size(0), -1, -1)
        encoder_output = self.encoder(image)
        memory = torch.cat([cls_token, encoder_output], dim=1)
        
        tgt_padding_mask = (tgt_tokens == self.pad_token_id)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(image.device)
        
        logits = self.decoder(memory, tgt_tokens, tgt_mask, tgt_padding_mask)
        
        return logits
class LLM_Vision_Model(nn.Module):
    """
    A model that connects our ViT ImageEncoder to a pre-trained LLM (Gemma).
    """
    def __init__(self, llm_model_name="google/gemma-2b"):
        super().__init__()
        
        # 1. Load the pre-trained LLM and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)

        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm.config.eos_token_id

        # 2. Freeze the LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 3. Use LoRA for parameter-efficient fine-tuning of the LLM
        target_modules = [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj", 
            "model.layers.0.mlp.down_proj"
        ]
        
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # 4. Create our ViT image encoder
        self.encoder = ImageEncoder() # Using the same ViT encoder
        
        # 5. Create the "connector" or projection layer
        # This layer maps the ViT's output dimension to the LLM's input dimension
        self.projection = nn.Linear(768, self.llm.config.hidden_size)

# In src/model.py, replace the LLM_Vision_Model class

# (Make sure these imports are at the top)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

class LLM_Vision_Model(nn.Module):
    """
    A model that connects our ViT ImageEncoder to a pre-trained LLM (Gemma).
    (This is the simpler version WITHOUT RAG)
    """
    def __init__(self, llm_model_name="stabilityai/stablelm-2-zephyr-1_6b"):
        super().__init__()
        
        # 1. Load LLM and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name,trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16,trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm.config.eos_token_id

        # 2. Freeze LLM and apply LoRA
        for param in self.llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        self.llm = get_peft_model(self.llm, lora_config)

        # 3. Load Vision Encoder and Projection Layer
        self.encoder = ImageEncoder()
        self.projection = nn.Linear(768, self.llm.config.hidden_size)
# In src/model.py, add this method inside the LLM_Vision_Model class

# In src/model.py, replace the generate method in the LLM_Vision_Model class

    def generate(self, image, max_len=256):
        """
        Generates a report for a given image with corrected loop logic.
        """
        self.eval()
        
        start_token_id = self.tokenizer.bos_token_id
        end_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            # 1. Encode the image and project features
            image_features = self.encoder(image)[:, 0, :]
            projected_features = self.projection(image_features).to(torch.bfloat16)
            
            # 2. Get the word embedding layer
            embedding_layer = self.llm.get_input_embeddings()
            
            # 3. Create the initial image embedding for the LLM
            # This is the context that the text will be generated from
            inputs_embeds = projected_features.unsqueeze(1)
            
            # 4. Start with the beginning-of-sequence token ID
            generated_ids = torch.full((image.size(0), 1), start_token_id, dtype=torch.long).to(image.device)

            for _ in range(max_len - 1):
                # 5. Get the embeddings for all currently generated tokens
                text_embeddings = embedding_layer(generated_ids)
                
                # 6. Combine image and text embeddings for the current step
                combined_embeddings = torch.cat([inputs_embeds, text_embeddings], dim=1)
                attention_mask = torch.ones(combined_embeddings.shape[:2], device=image.device)
                
                # 7. Get model predictions
                outputs = self.llm(inputs_embeds=combined_embeddings, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] # We only care about the very last token's prediction
                
                # 8. Get the next token (greedy search)
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                
                # 9. Append the new token ID for the next iteration
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 10. Stop if we generate the end-of-sequence token
                if (next_token == end_token_id).all():
                    break
                    
        return generated_ids

    def forward(self, image, tgt_tokens):
        # Get image features from the encoder's [CLS] token
        image_features = self.encoder(image)[:, 0, :]
        
        # Project image features to the LLM's dimension
        projected_features = self.projection(image_features).to(torch.bfloat16)
        embedding_layer = self.llm.get_input_embeddings()
        # Get word embeddings for the target text
        text_embeddings = embedding_layer(tgt_tokens)
        
        # Combine the image and text embeddings
        combined_embeddings = torch.cat([projected_features.unsqueeze(1), text_embeddings], dim=1)
        
        # Create a simple attention mask
        attention_mask = torch.ones(combined_embeddings.size(0), combined_embeddings.size(1)).to(image.device)
        
        # Pass the combined sequence to the LLM
        outputs = self.llm(inputs_embeds=combined_embeddings, attention_mask=attention_mask)

        # We only care about the loss for the text part of the sequence
        logits = outputs.logits[:, 1:, :] # Skip the logit for the image feature
        return logits   
