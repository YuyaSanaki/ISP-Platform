import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

def compute_mean_embs(hidden_state, length, max_len):
    # hidden_state: (batch, seq_len, hidden_dim)
    # length: (batch,)
    batch_size = hidden_state.size(0)
    device = hidden_state.device
    mask = torch.arange(max_len, device=device).unsqueeze(0) < length.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(hidden_state).float()
    masked_embs = hidden_state * mask
    mean_embs = masked_embs.sum(1) / length.view(-1, 1).float()
    return mean_embs

def extract_embeddings(model, dataset, device, batch_size=50):
    all_embs = []
    labels = []
    model.eval()

    
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # dynamic padding
        from geneformer import tokenizer as gf_tokenizer
        with open(gf_tokenizer.TOKEN_DICTIONARY_FILE, 'rb') as f:
            t_dict = pickle.load(f)
        pad_token_id = t_dict.get("<pad>", 0)
        
        max_len = max(batch["length"])
        padded_input_ids = []
        padded_attention_masks = []
        for input_id_list in batch["input_ids"]:
            length = len(input_id_list)
            padded_input_ids.append(input_id_list + [pad_token_id] * (max_len - length))
            padded_attention_masks.append([1] * length + [0] * (max_len - length))
            
        input_ids = torch.tensor(padded_input_ids).to(device)
        attention_mask = torch.tensor(padded_attention_masks).to(device)
        lengths = torch.tensor(batch["length"]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # using last hidden state (emb_layer = 0 in Geneformer terminology means last layer, but index is -1)
            # Wait, hidden_states[-1] is the last layer.
            hidden_states = outputs.hidden_states[-1]
            embs = compute_mean_embs(hidden_states, lengths, max_len=input_ids.shape[1])
            all_embs.append(embs.cpu().numpy())
            
    return np.concatenate(all_embs, axis=0)

def main():
    print("Loading model and dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths from the config
    geneformer_model_path = "/app/output/20260410/finetune_075259_676389Z/all_run1"
    dataset_path = "/app/data/PIPseq/tokenized_dataset/PIPseq_0.dataset"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        geneformer_model_path,
        num_labels=2,
        output_hidden_states=True
    ).to(device)
    
    dataset = load_from_disk(dataset_path)
    
    # Get token for Igfbp2
    from geneformer import tokenizer as gf_tokenizer
    with open(gf_tokenizer.TOKEN_DICTIONARY_FILE, 'rb') as f:
        token_dict = pickle.load(f)
    ensembl_id = "ENSMUSG00000039323" # Igfbp2
    igfbp2_token = token_dict.get(ensembl_id)
    if not igfbp2_token:
        print(f"Token for {ensembl_id} not found.")
        return
    print(f"Igfbp2 token: {igfbp2_token}")
    
    # Filter dataset for AD and WT *then* limit
    ad_dataset = dataset.filter(lambda x: x["disease"] == "AD")
    wt_dataset = dataset.filter(lambda x: x["disease"] == "WT")
    
    max_cells = 2000
    ad_dataset = ad_dataset.select(range(min(len(ad_dataset), max_cells)))
    wt_dataset = wt_dataset.select(range(min(len(wt_dataset), max_cells)))
    
    print(f"Found {len(ad_dataset)} AD cells and {len(wt_dataset)} WT cells.")
    
    def perturb_dataset(batch):
        new_input_ids = []
        new_lengths = []
        new_attention_masks = []
        for input_id_list in batch["input_ids"]:
            # Delete igfbp2_token
            perturbed_ids = [tok for tok in input_id_list if tok != igfbp2_token]
            length = len(perturbed_ids)
            attn_mask = [1]*length
            new_input_ids.append(perturbed_ids)
            new_lengths.append(length)
            new_attention_masks.append(attn_mask)
        return {"input_ids": new_input_ids, "length": new_lengths, "attention_mask": new_attention_masks}

    ad_perturbed_dataset = ad_dataset.map(perturb_dataset, batched=True, batch_size=50)
    
    print("Extracting WT embeddings...")
    wt_embs = extract_embeddings(model, wt_dataset, device)
    
    print("Extracting AD embeddings...")
    ad_embs = extract_embeddings(model, ad_dataset, device)
    
    print("Extracting AD (perturbed) embeddings...")
    ad_pert_embs = extract_embeddings(model, ad_perturbed_dataset, device)
    
    all_embs = np.vstack([wt_embs, ad_embs, ad_pert_embs])
    labels = ["WT"] * len(wt_embs) + ["AD"] * len(ad_embs) + ["AD+ISP(Igfbp2)"] * len(ad_pert_embs)
    
    print("Running CPU UMAP...")
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    umap_embs = reducer.fit_transform(all_embs)
    
    df = pd.DataFrame(umap_embs, columns=["UMAP 1", "UMAP 2"])
    df["State"] = labels
    
    out_dir = "/app/output/20260413/run_012117_080951Z/figures/"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="UMAP 1", y="UMAP 2", hue="State", alpha=0.7, 
                    palette={"WT": "steelblue", "AD": "coral", "AD+ISP(Igfbp2)": "red"})
    plt.title("UMAP of Cell Embeddings: WT vs AD vs AD+ISP(Igfbp2)")
    
    # draw arrows from AD to AD+ISP
    ad_idx = len(wt_embs)
    ad_pert_idx = len(wt_embs) + len(ad_embs)
    
    # draw a subset of arrows to avoid cluttering
    step = max(1, len(ad_embs) // 100) # draw ~100 arrows
    for i in range(0, len(ad_embs), step):
        plt.arrow(umap_embs[ad_idx+i, 0], umap_embs[ad_idx+i, 1],
                  umap_embs[ad_pert_idx+i, 0] - umap_embs[ad_idx+i, 0],
                  umap_embs[ad_pert_idx+i, 1] - umap_embs[ad_idx+i, 1],
                  color='gray', alpha=0.3, width=0.01, head_width=0.1)

    out_file = os.path.join(out_dir, "umap_AD_vs_WT_isp_Igfbp2.png")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved UMAP to {out_file}")

if __name__ == "__main__":
    main()
