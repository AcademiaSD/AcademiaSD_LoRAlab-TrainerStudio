# -*- coding: utf-8 -*-
"""Pre-cache ERNIE-Image text embeddings and FLUX.2-format VAE latents."""
import gc, json, math, os, sys
import torch
import torchvision.transforms.functional as VF
from PIL import Image
from diffusers import AutoencoderKLFlux2, ErnieImagePipeline
from transformers import AutoTokenizer, Mistral3Model

CONFIG = "settings/pre_cache_settings_ernie.json"
DEFAULTS = {"model_id": "Ernie-NF4", "dataset_path": "./dataset", "cache_dir": "./cached_data_ernie", "target_area": 512*512, "max_side": 1536, "multiple": 32, "project_name": "", "trigger_word": "", "preview_custom_prompt": ""}
cfg = json.load(open(CONFIG, encoding="utf-8")) if os.path.isfile(CONFIG) else {}
MODEL_ID = cfg.get("model_id", DEFAULTS["model_id"])
DATASET = cfg.get("dataset_path", DEFAULTS["dataset_path"])
PROJECT = cfg.get("project_name", "").strip()
CACHE = f"./cached_data_ernie_{PROJECT}" if PROJECT else cfg.get("cache_dir", DEFAULTS["cache_dir"])
AREA = int(cfg.get("target_area", DEFAULTS["target_area"]))
MAX_SIDE = int(cfg.get("max_side", DEFAULTS["max_side"]))
MULTIPLE = 32
TRIGGER = cfg.get("trigger_word", "").strip()
CUSTOM = cfg.get("preview_custom_prompt", "").strip()
os.makedirs(CACHE, exist_ok=True)

def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def bucket(w, h):
    ar = w / h
    bh = math.sqrt(AREA / ar); bw = ar * bh
    bw = max(MULTIPLE, round(bw/MULTIPLE)*MULTIPLE); bh = max(MULTIPLE, round(bh/MULTIPLE)*MULTIPLE)
    if max(bw,bh) > MAX_SIDE:
        scale = MAX_SIDE/max(bw,bh); bw=max(MULTIPLE,int(bw*scale)//MULTIPLE*MULTIPLE); bh=max(MULTIPLE,int(bh*scale)//MULTIPLE*MULTIPLE)
    return bw,bh

def with_trigger(text):
    return f"{TRIGGER}, {text}" if TRIGGER and TRIGGER.lower() not in text.lower() else text

def ensure_nf4_model():
    converted=(os.path.join(MODEL_ID,"transformer","index.json"),
               os.path.join(MODEL_ID,"transformer","others.safetensors"),
               os.path.join(MODEL_ID,"text_encoder","config.json"),
               os.path.join(MODEL_ID,"text_encoder","model.safetensors"),
               os.path.join(MODEL_ID,"vae","diffusion_pytorch_model.safetensors"),
               os.path.join(MODEL_ID,"tokenizer","tokenizer.json"))
    if all(os.path.isfile(p) for p in converted):
        return
    from huggingface_hub import snapshot_download
    repo_id="AcademiaSD/Ernie-NF4-for-LoRA-Training"
    print(f"Downloading the prepared ERNIE NF4 model from Hugging Face ({repo_id}).",flush=True)
    os.makedirs(MODEL_ID,exist_ok=True)
    snapshot_download(repo_id=repo_id,local_dir=MODEL_ID)
    missing=[p for p in converted if not os.path.isfile(p)]
    if missing:
        raise RuntimeError("The ERNIE NF4 repository download is incomplete; missing: "+", ".join(missing))

def encode_text(te, tok, prompt, device):
    ids = tok(prompt, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
    if not ids: ids = [tok.bos_token_id or 0]
    x = torch.tensor([ids], device=device)
    with torch.inference_mode():
        hidden = te(input_ids=x, output_hidden_states=True).hidden_states[-2][0]
    return hidden.to(torch.bfloat16).cpu()

def patchify(z):
    return ErnieImagePipeline._patchify_latents(z)

def fit(img, w, h):
    r=max(w/img.width,h/img.height); img=img.resize((math.ceil(img.width*r), math.ceil(img.height*r)), Image.Resampling.LANCZOS)
    x=(img.width-w)//2; y=(img.height-h)//2
    return img.crop((x,y,x+w,y+h))

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for pre-cache.")
    if not os.path.isdir(DATASET): raise FileNotFoundError(f"Dataset folder not found: {DATASET}")
    ensure_nf4_model()
    files=sorted(f for f in os.listdir(DATASET) if f.lower().endswith((".png",".jpg",".jpeg",".webp")))
    samples=[]
    for f in files:
        stem=os.path.splitext(f)[0]
        # ERNIE is text-to-image. These files are independent images; there is no paired edit path.
        cap=os.path.join(DATASET,stem+".txt")
        prompt=open(cap,encoding="utf-8").read().strip() if os.path.isfile(cap) else ""
        samples.append((stem,os.path.join(DATASET,f),with_trigger(prompt)))
    if not samples: raise RuntimeError(f"No images found in {DATASET}")
    print(f"ERNIE-Image text-to-image pre-cache: {len(samples)} images | {CACHE}",flush=True)
    expected={f"{n}_{s}.pt" for n,_,_ in samples for s in ("latent","embed")}
    expected|={"_neg_embed.pt"}
    if CUSTOM: expected|={"_custom_embed.pt","_custom_prompt.txt"}
    for f in os.listdir(CACHE):
        if f.endswith(("_latent.pt","_embed.pt")) and not f.startswith("_") and f not in expected: os.remove(os.path.join(CACHE,f))
    te=Mistral3Model.from_pretrained(os.path.join(MODEL_ID,"text_encoder"),torch_dtype=torch.bfloat16,device_map="cuda").eval()
    tok=AutoTokenizer.from_pretrained(os.path.join(MODEL_ID,"tokenizer"))
    for i,(name,path,prompt) in enumerate(samples,1):
        torch.save(encode_text(te,tok,prompt,"cuda"),os.path.join(CACHE,name+"_embed.pt"))
        print(f"[{i}/{len(samples)}] text: {name}",flush=True)
    torch.save(encode_text(te,tok,"","cuda"),os.path.join(CACHE,"_neg_embed.pt"))
    if CUSTOM:
        cp=with_trigger(CUSTOM); torch.save(encode_text(te,tok,cp,"cuda"),os.path.join(CACHE,"_custom_embed.pt"))
        open(os.path.join(CACHE,"_custom_prompt.txt"),"w",encoding="utf-8").write(cp)
        open(os.path.join(CACHE,"_custom_image.txt"),"w",encoding="utf-8").write("")
    del te,tok; free()
    vae=AutoencoderKLFlux2.from_pretrained(MODEL_ID,subfolder="vae",torch_dtype=torch.bfloat16).to("cuda").eval()
    mean=vae.bn.running_mean.view(1,-1,1,1).to("cuda",torch.float32)
    std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+1e-5).to("cuda",torch.float32)
    for i,(name,path,_) in enumerate(samples,1):
        out=os.path.join(CACHE,name+"_latent.pt")
        if os.path.isfile(out) and os.path.getmtime(out)>os.path.getmtime(path):
            try:
                if torch.load(out,weights_only=True).shape[-2:]==(bucket(*Image.open(path).size)[1]//32,bucket(*Image.open(path).size)[0]//32): continue
            except Exception: pass
        w,h=bucket(*Image.open(path).size)
        with Image.open(path) as im: im=fit(im.convert("RGB"),w,h)
        px=(VF.pil_to_tensor(im).unsqueeze(0).float()/127.5-1).to("cuda",torch.bfloat16)
        with torch.inference_mode(): z=vae.encode(px).latent_dist.mode().float(); lat=((patchify(z)-mean)/std).to(torch.bfloat16)
        tmp=out+".tmp"; torch.save(lat.cpu(),tmp); os.replace(tmp,out)
        print(f"[{i}/{len(samples)}] latent: {name} {w}x{h}",flush=True)
        del px,z,lat
    del vae; free(); print("Pre-cache complete.",flush=True)

def encode_preview(cache,prompt,image,device):
    te=Mistral3Model.from_pretrained(os.path.join(MODEL_ID,"text_encoder"),torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,device_map=device).eval()
    tok=AutoTokenizer.from_pretrained(os.path.join(MODEL_ID,"tokenizer"))
    prompt=with_trigger(prompt); emb=encode_text(te,tok,prompt,device)
    torch.save(emb,os.path.join(cache,"_custom_embed.pt")); open(os.path.join(cache,"_custom_prompt.txt"),"w",encoding="utf-8").write(prompt)
    open(os.path.join(cache,"_custom_image.txt"),"w",encoding="utf-8").write("")
    print("[Custom Prompt] Ready.",flush=True)

if __name__=="__main__":
    if len(sys.argv)==6 and sys.argv[1]=="--prompt-only": encode_preview(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    else: main()
