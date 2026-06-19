import random
from typing import List, Optional

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, GPT2Model
from aggregator.ABMIL import AttentionGated

################################################################################
# Shared trunk + 2 heads
################################################################################
class MultiHeadGPT2(nn.Module):
    def __init__(self, model_path: str = "gpt2", device: str = "cuda"):
        super().__init__()
        self.device = device
        config = AutoConfig.from_pretrained(model_path)
        self.config = config
        self.shared_trunk = GPT2Model.from_pretrained(model_path, config=config)

        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        self.coarse_head = GPT2Block(config)
        self.fine_head = GPT2Block(config)

        vocab_size = config.vocab_size
        hidden_size = config.n_embd
        self.coarse_unembedding = nn.Linear(hidden_size, vocab_size, bias=False)
        self.fine_unembedding = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        coarse_out = self.coarse_head(hidden_states, attention_mask=attention_mask)[0]
        fine_out = self.fine_head(hidden_states, attention_mask=attention_mask)[0]
        coarse_logits = self.coarse_unembedding(coarse_out)
        fine_logits = self.fine_unembedding(fine_out)
        return coarse_logits, fine_logits


class TaxoMIL(nn.Module):
    def __init__(
        self,
        model_path: str = "gpt2",
        device: str = "cuda",
        decoder_type: str = "GPT2",
        output_len_coarse: int = 5,
        output_len_fine: int = 15,
        all_coarse_labels: Optional[List[str]] = None,
        all_fine_labels: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        emb_dim: int = 1024,
        gated: bool = True,
        prompt_templates: Optional[List[str]] = None,
    ):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.output_len_coarse = output_len_coarse
        self.output_len_fine = output_len_fine
        self.all_coarse_labels = all_coarse_labels
        self.all_fine_labels = all_fine_labels
        self.gated = gated

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        if self.tokenizer.pad_token is None and decoder_type in ["GPT2", "BioGPT"]:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        self.hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        # Decoder
        self.decoder = MultiHeadGPT2(model_path=model_path, device=device)
        if self.tokenizer.pad_token is not None and decoder_type in ["GPT2", "BioGPT"]:
            self.decoder.shared_trunk.resize_token_embeddings(len(self.tokenizer))

        out_dim = self.emb_dim
        self.aggregator = AttentionGated(input_dim=out_dim, gated=self.gated, dropout=0.25).to(self.device)

        # 512/512 split adaptors
        in_features = out_dim // 2
        self.coarse_adaptor = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        ).to(self.device)

        self.fine_adaptor = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        ).to(self.device)

        if self.all_coarse_labels is not None and self.all_fine_labels is not None:
            self.text_projector = nn.Linear(self.hidden_size, self.hidden_size).to(device)

        # Image token
        self.image_token = "<|img|>"
        self.image_token_embed = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, dtype=torch.float32, device=self.device)
        )

        to_add = []
        vocab = self.tokenizer.get_vocab()
        if self.image_token not in vocab:
            to_add.append(self.image_token)

        if len(to_add) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            self.decoder.shared_trunk.resize_token_embeddings(len(self.tokenizer))

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        base_prompt_templates = prompt_templates or self._get_default_prompts()
        domain = self._get_dataset_domain(dataset)
        self.training_prompts = [p.format(domain=domain) if "{domain}" in p else p for p in base_prompt_templates]

        if self.tokenizer.bos_token_id is None:
            raise ValueError("tokenizer.bos_token_id is None (bos fallback disabled).")

    # ========== Static Helpers ==========
    @staticmethod
    def _get_default_prompts() -> List[str]:
        return [
            "The observed condition in the {domain} image is: ",
            "The diagnosis for the {domain} image shows signs of: ",
            "The medical findings for this image indicate: ",
            "This {domain} image reveals the presence of: ",
            "The condition detected in the {domain} image is: ",
            "The histopathological observation in this image suggests: ",
            "The pathology result from this {domain} image describes: ",
            "The condition diagnosed from the image analysis is: ",
            "The elements detected in the {domain} region of the image are: ",
            "The visible medical observation in the image corresponds to: ",
        ]

    @staticmethod
    def _get_dataset_domain(dataset: Optional[str]) -> str:
        ds2domain = {"BRACS": "breast", "PANDA": "prostate"}
        return ds2domain.get(str(dataset).upper(), "gastric")

    # ========== Text Embedding Helpers ==========
    def _get_mean_pooled_text_embeddings(self, labels: List[str]) -> torch.Tensor:
        text_encoder = self.decoder.shared_trunk.wte
        tokenized = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        embeddings = text_encoder(tokenized.input_ids)
        mask = tokenized.attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def get_context_label_embeddings(self, head: str) -> torch.Tensor:
        if head == "coarse" and self.all_coarse_labels is not None:
            coarse_embeds = self._get_mean_pooled_text_embeddings(self.all_coarse_labels)
            label_context = coarse_embeds.mean(dim=0)
            return label_context
        if head == "fine" and self.all_fine_labels is not None:
            fine_embeds = self._get_mean_pooled_text_embeddings(self.all_fine_labels)
            label_context = fine_embeds.mean(dim=0)
            return label_context
        return torch.zeros(self.hidden_size, device=self.device)

    # ========== Prefix Building Helpers ==========
    def _build_prefix_embeds(self, visual_embeds: torch.Tensor, text_prompt: str, head: str):
        B = visual_embeds.size(0)
        visual_embeds = visual_embeds.unsqueeze(1)
        image_token_embed = self.image_token_embed.repeat(B, 1, 1)

        bos_id = self.tokenizer.bos_token_id
        embedding_layer = self.decoder.shared_trunk.wte

        prompt_tokens = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            padding=isinstance(text_prompt, list),
            add_special_tokens=False,
        ).to(self.device)

        prompt_ids = prompt_tokens["input_ids"]
        bos_tensor = torch.full((prompt_ids.size(0), 1), bos_id, device=self.device, dtype=torch.long)
        prompt_ids = torch.cat([bos_tensor, prompt_ids], dim=1)
        prompt_embeds = embedding_layer(prompt_ids)
        if prompt_embeds.size(0) != B:
            prompt_embeds = prompt_embeds.repeat(B, 1, 1)

        label_context = self.get_context_label_embeddings(head).unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)
        prefix = torch.cat([visual_embeds, image_token_embed, label_context, prompt_embeds], dim=1)
        return prefix

    # ========== Head Helpers ==========
    def _get_head_and_unembedding(self, head: str):
        if head == "coarse":
            return self.decoder.coarse_head, self.decoder.coarse_unembedding
        else:
            return self.decoder.fine_head, self.decoder.fine_unembedding

    def _head_step(self, head: str, trunk_hs: torch.Tensor, head_past=None):
        head_layer, unembedding = self._get_head_and_unembedding(head)
        out = head_layer(trunk_hs, layer_past=head_past, use_cache=True)
        hs, new_past = out[0], out[1]
        logits_last = unembedding(hs[:, -1:, :])
        return logits_last, new_past

    # ========== Training Helpers ==========
    def _teacher_force_head(self, base_embeds: torch.Tensor, target_ids: torch.LongTensor, head: str):
        B, Lbase, H = base_embeds.shape
        T = target_ids.size(1)

        y_in = target_ids[:, :-1]
        y_in_emb = self.decoder.shared_trunk.wte(y_in)
        inputs = torch.cat([base_embeds, y_in_emb], dim=1)

        attn = torch.ones(inputs.size()[:2], dtype=torch.long, device=self.device)
        pos = torch.arange(0, inputs.size(1), device=self.device).unsqueeze(0).expand(B, -1)

        trunk = self.decoder.shared_trunk(
            inputs_embeds=inputs,
            attention_mask=attn,
            position_ids=pos,
            use_cache=False,
            return_dict=True,
        )
        hs = trunk.last_hidden_state

        head_layer, unembedding = self._get_head_and_unembedding(head)
        hd = head_layer(hs)[0]
        logits = unembedding(hd)

        gen_logits = logits[:, (Lbase - 1):(Lbase - 1 + T), :]
        return gen_logits

    # ========== Generation Helpers ==========
    def _generate_cached(self, prefix_embeds: torch.Tensor, gen_len: int, head: str):
        B = prefix_embeds.size(0)
        embedding_layer = self.decoder.shared_trunk.wte

        prefix_len = prefix_embeds.size(1)
        attn_mask = torch.ones((B, prefix_len), dtype=torch.long, device=self.device)
        pos_ids = torch.arange(0, prefix_len, device=self.device).unsqueeze(0).expand(B, -1)

        trunk_out = self.decoder.shared_trunk(
            inputs_embeds=prefix_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        trunk_hs = trunk_out.last_hidden_state
        trunk_past = trunk_out.past_key_values

        logits_last, head_past = self._head_step(head, trunk_hs, head_past=None)
        next_tokens = torch.argmax(logits_last, dim=-1)

        generated_tokens = [next_tokens]
        logits_list = [logits_last]
        total_len = prefix_len + 1

        for _ in range(gen_len - 1):
            new_emb = embedding_layer(next_tokens)  # (B,1,H)

            attn_mask = torch.ones((B, total_len), dtype=torch.long, device=self.device)
            pos_id = torch.full((B, 1), total_len - 1, device=self.device, dtype=torch.long)

            trunk_out = self.decoder.shared_trunk(
                inputs_embeds=new_emb,
                attention_mask=attn_mask,
                position_ids=pos_id,
                past_key_values=trunk_past,
                use_cache=True,
                return_dict=True,
            )
            trunk_last = trunk_out.last_hidden_state
            trunk_past = trunk_out.past_key_values

            logits_last, head_past = self._head_step(head, trunk_last, head_past=head_past)
            next_tokens = torch.argmax(logits_last, dim=-1)

            generated_tokens.append(next_tokens)
            logits_list.append(logits_last)
            total_len += 1

        generated_tokens = torch.cat(generated_tokens, dim=1)
        all_logits = torch.cat(logits_list, dim=1)

        final_sequences = []
        for seq in generated_tokens:
            seq_list = seq.tolist()
            if self.tokenizer.eos_token_id in seq_list:
                idx = seq_list.index(self.tokenizer.eos_token_id)
                seq_list = seq_list[: idx + 1]
            final_sequences.append(seq_list)

        decoded_texts = [self.tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in final_sequences]
        return decoded_texts, all_logits

    # ===========================================================
    def encode_image_features(self, image_embedding: torch.Tensor):
        agg_out = self.aggregator(image_embedding).squeeze(1)
        split_dim = agg_out.shape[1] // 2
        coarse = agg_out[:, :split_dim]
        fine = agg_out[:, split_dim:]
        coarse_vis = self.coarse_adaptor(coarse)
        fine_vis = self.fine_adaptor(fine)
        return coarse_vis, fine_vis

    def forward_train(self, image_embedding: torch.Tensor, coarse_tgt_ids: torch.LongTensor, fine_tgt_ids: torch.LongTensor):
        prompt = random.choice(self.training_prompts) if self.training else self.training_prompts[0]
        coarse_vis, fine_vis = self.encode_image_features(image_embedding)
        base_c = self._build_prefix_embeds(coarse_vis, prompt, head="coarse")
        base_f = self._build_prefix_embeds(fine_vis, prompt, head="fine")
        c_logits = self._teacher_force_head(base_c, coarse_tgt_ids, head="coarse")
        f_logits = self._teacher_force_head(base_f, fine_tgt_ids, head="fine")
        return {"coarse_logits": c_logits, "fine_logits": f_logits}

    def forward_coarse(self, image_embedding: torch.Tensor, prompt: str):
        coarse_vis, _ = self.encode_image_features(image_embedding)
        prefix = self._build_prefix_embeds(coarse_vis, prompt, head="coarse")
        return self._generate_cached(prefix, self.output_len_coarse, head="coarse")

    def forward_fine(self, image_embedding: torch.Tensor, prompt: str):
        _, fine_vis = self.encode_image_features(image_embedding)
        prefix = self._build_prefix_embeds(fine_vis, prompt, head="fine")
        return self._generate_cached(prefix, self.output_len_fine, head="fine")

    def forward(self, image_embedding: torch.Tensor):
        prompt = random.choice(self.training_prompts) if self.training else self.training_prompts[0]
        coarse_texts, coarse_logits = self.forward_coarse(image_embedding, prompt)
        fine_texts, fine_logits = self.forward_fine(image_embedding, prompt)
        return coarse_texts, fine_texts, coarse_logits, fine_logits


def build_model(
    model_path: str = "gpt2",
    device: str = "cuda",
    decoder_type: str = "GPT2",
    dataset: Optional[str] = None,
    emb_dim: int = 1024,
    gated: bool = True,
    # labels
    all_coarse_labels: Optional[List[str]] = None,
    all_fine_labels: Optional[List[str]] = None,
    # lengths
    output_len_coarse: int = 5,
    output_len_fine: int = 15,
    # prompts
    prompt_templates: Optional[List[str]] = None,
) -> TaxoMIL:
    return TaxoMIL(
        model_path=model_path,
        device=device,
        decoder_type=decoder_type,
        output_len_coarse=output_len_coarse,
        output_len_fine=output_len_fine,
        all_coarse_labels=all_coarse_labels,
        all_fine_labels=all_fine_labels,
        dataset=dataset,
        emb_dim=emb_dim,
        gated=gated,
        prompt_templates=prompt_templates,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fake_image_embed = torch.randn(2, 10, 1024, device=device)

    m = build_model(
        model_path="gpt2",
        device=device,
        dataset="BRACS",
        emb_dim=1024,
        gated=True,
        output_len_coarse=5,
        output_len_fine=15,
        all_coarse_labels=["benign", "malignant"],
        all_fine_labels=["adenosis", "invasive"],
    ).to(device)
    m.eval()
    with torch.no_grad():
        out = m(fake_image_embed)
        print("[OK] :", type(out), len(out))