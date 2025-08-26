# src/llm_model.py
import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from src.llm_config import TOKENIZER_MODEL, IMAGE_MODEL, IMAGE_EMBED_DIM

class ImageEncoder(nn.Module):
    # ... (this part is correct, no changes needed) ...
    def __init__(self, model_name=IMAGE_MODEL, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        original_conv = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        self.model.patch_embed.proj.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    def forward(self, x):
        return self.model.forward_features(x)


class LLM_Vision_Model(nn.Module):
    # ... (__init__ is correct, no changes needed) ...
    def __init__(self, llm_model_name=TOKENIZER_MODEL):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.tokenizer.eos_token_id
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.encoder = ImageEncoder()
        self.projection = nn.Linear(IMAGE_EMBED_DIM, self.llm.config.hidden_size)


    # --- CHANGE IS HERE ---
    # Add 'attention_mask' as the third argument
    def forward(self, image, tgt_tokens, attention_mask):
        image_features = self.encoder(image)
        projected_features = self.projection(image_features).to(torch.bfloat16)
        
        embedding_layer = self.llm.get_input_embeddings()
        text_embeddings = embedding_layer(tgt_tokens)
        
        inputs_embeds = torch.cat([projected_features, text_embeddings], dim=1)
        
        # Create a combined attention mask for the image features and text tokens
        image_attention_mask = torch.ones(projected_features.shape[:2], dtype=torch.long, device=image.device)
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # Pass the combined mask to the underlying LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=combined_attention_mask)

        # Slice the logits to only return predictions for the text part
        logits = outputs.logits[:, projected_features.size(1):, :]
        return logits

    # ... (generate method is correct, no changes needed) ...
    def generate(self, image, max_len=256):
     self.eval()
     with torch.no_grad():
        image_features = self.encoder(image)
        projected_features = self.projection(image_features).to(torch.bfloat16)
        
        attention_mask = torch.ones(projected_features.shape[:2], dtype=torch.long, device=image.device)

        # Add the same prefix used during training
        prefix_text = "Generate a chest X-ray report: "
        prefix_tokens = self.tokenizer(prefix_text, return_tensors='pt', 
                                     add_special_tokens=False).input_ids.to(image.device)
        prefix_embeds = self.llm.get_input_embeddings()(prefix_tokens)
        
        # Combine image features with prefix embeddings
        combined_embeds = torch.cat([projected_features, prefix_embeds], dim=1)
        combined_attention = torch.ones(combined_embeds.shape[:2], dtype=torch.long, 
                                      device=image.device)

        output_ids = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            max_new_tokens=max_len,
            num_beams=5,        # Use beam search for better coherence
            early_stopping=True,
            temperature=0.7,    # Reduced randomness
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        # Remove the prefix tokens from the output
        output_ids = output_ids[:, prefix_tokens.size(1):]
        return output_ids