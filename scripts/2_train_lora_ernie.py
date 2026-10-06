# -*- coding: utf-8 -*-
"""NF4 LoRA training for ERNIE-Image (text-to-image only)."""
import gc, json, math, os, random, signal, sys, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from diffusers import AutoencoderKLFlux2, ErnieImagePipeline, ErnieImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load, load_file, save_file
import lora_options

# La barra de progreso comparte los caracteres visuales del entrenador Klein. Fuerza UTF-8
# también cuando Python se lanza desde una consola Windows con codepage cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    "█░".encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    PROGRESS_FILLED, PROGRESS_EMPTY = "█", "░"
except (UnicodeEncodeError, LookupError):
    # Consolas embebidas pueden rechazar reconfigure(); nunca dejes que la barra pare el trainer.
    PROGRESS_FILLED, PROGRESS_EMPTY = "#", "-"

CONFIG="settings/train_settings_ernie.json"
cfg=json.load(open(CONFIG,encoding="utf-8")) if os.path.isfile(CONFIG) else {}
MODEL=cfg.get("model_id","Ernie-NF4"); PROJECT=cfg.get("project_name","").strip()
CACHE=f"./cached_data_ernie_{PROJECT}" if PROJECT else cfg.get("cache_dir","./cached_data_ernie")
OUT=f"./ernie_lora_output_{PROJECT}" if PROJECT else cfg.get("output_dir","./ernie_lora_output")
STEPS=int(cfg.get("total_steps",1000)); BATCH=int(cfg.get("batch_size",1)); ACCUM=max(1,int(cfg.get("grad_accum_steps",4)))
LR=float(cfg.get("lr",3e-4)); WARMUP=int(cfg.get("warmup_steps",100)); MIN_LR_RATIO=float(cfg.get("min_lr_ratio",.1)); SAVE_EVERY=int(cfg.get("save_every",100))
RANK=int(cfg.get("lora_rank",16)); ALPHA=float(cfg.get("lora_alpha",16)); USE_RSLORA=bool(cfg.get("use_rslora",False)); USE_LORAPLUS=bool(cfg.get("use_loraplus",False)); LORAPLUS_RATIO=float(cfg.get("loraplus_ratio",16)); USE_LOKR=bool(cfg.get("use_lokr",False)); LOKR_FACTOR=max(1,int(cfg.get("lokr_factor",16)))
LORA_TARGETS="all" if cfg.get("lora_targets","blocks")=="all" else "blocks"
WEIGHT_DECAY=float(cfg.get("weight_decay",0)); MAX_GRAD_NORM=float(cfg.get("max_grad_norm",1.0)); PREVIEW_EVERY=int(cfg.get("preview_every",0))
PREVIEW_STEPS=int(cfg.get("preview_steps",28)); PREVIEW_CFG=float(cfg.get("preview_cfg",4)); PREVIEW_SIZE=int(cfg.get("preview_size",0))
PREVIEW_MODE=cfg.get("preview_caption_mode","first")
SEED=int(cfg.get("seed",42)); TRIGGER=cfg.get("trigger_word","").strip(); CUSTOM=cfg.get("preview_custom_prompt","").strip()
os.makedirs(OUT,exist_ok=True); RESUME=os.path.join(OUT,"resume_checkpoint"); OPT=os.path.join(OUT,"optimizer.pt"); STEPFILE=os.path.join(OUT,"current_step.txt")
torch.manual_seed(SEED if SEED>0 else random.randrange(2**31)); random.seed(SEED if SEED>0 else None)

def free(): gc.collect(); torch.cuda.empty_cache()

def load_nf4(model_dir):
    root=os.path.join(model_dir,"transformer"); index=json.load(open(os.path.join(root,"index.json"),encoding="utf-8")); weights=os.path.join(root,"weights")
    with init_empty_weights(): model=ErnieImageTransformer2DModel.from_config(ErnieImageTransformer2DModel.load_config(root))
    def put(name,layer):
        p,_,c=name.rpartition("."); setattr(model.get_submodule(p) if p else model,c,layer)
    for name,info in index["quantized"].items():
        path=os.path.join(weights,info["file"])
        with safe_open(path,framework="pt",device="cpu") as f:
            wd=f.get_tensor("weight"); qs={k[len("quant_state."):]:f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}; bias=f.get_tensor("bias") if "bias" in f.keys() else None
        w=Params4bit(wd,requires_grad=False,quant_type="nf4",quant_storage=torch.uint8); w.quant_state=QuantState.from_dict(qs,device="cpu"); w.bnb_quantized=True
        with torch.device("meta"): layer=Linear4bit(info["in_features"],info["out_features"],bias=info["bias"],quant_type="nf4",compute_dtype=torch.bfloat16)
        layer.weight=w
        if bias is not None: layer.bias=torch.nn.Parameter(bias,requires_grad=False)
        put(name,layer)
    model.load_state_dict(load_file(os.path.join(root,index["others"])),strict=False,assign=True)
    missing=[n for n,p in list(model.named_parameters())+list(model.named_buffers()) if p.is_meta]
    if missing: raise RuntimeError(f"Incomplete ERNIE NF4 transformer cache: {missing[:8]}")
    print(f"NF4 transformer loaded: {len(index['quantized'])} layers",flush=True)
    return model

def export(model,path,step):
    if USE_LOKR:
        save_file(lora_options.lokr_export(model,lambda name:"transformer."+name,lora_options.export_alpha(ALPHA,RANK,False)),path,metadata={"format":"pt","trained_with":"AcademiaSD LoRAlab ERNIE-Image","ss_sd_model_name":"baidu/ERNIE-Image","ss_steps":str(step)})
        return
    state={}
    for k,v in get_peft_model_state_dict(model).items():
        key="transformer."+k.replace("base_model.model.","")
        state[key]=v.to(torch.bfloat16).cpu().contiguous()
        if key.endswith(".lora_A.weight"): state[key[:-len(".lora_A.weight")]+".alpha"]=torch.tensor(float(lora_options.export_alpha(ALPHA,RANK,USE_RSLORA)),dtype=torch.float32)
    trigger=TRIGGER or ""
    meta={"format":"pt","trained_with":"AcademiaSD LoRAlab ERNIE-Image","ss_sd_model_name":"baidu/ERNIE-Image","ss_base_model_version":"ERNIE-Image","ss_network_module":"peft.LoraModel","ss_network_dim":str(RANK),"ss_network_alpha":str(lora_options.export_alpha(ALPHA,RANK,USE_RSLORA)),"ss_steps":str(step),"ss_max_train_steps":str(STEPS),"ss_learning_rate":str(LR),"ss_mixed_precision":"bf16","ss_optimizer":"AdamW","ss_weight_decay":str(WEIGHT_DECAY)}
    if trigger: meta["trigger_word"]=trigger; meta["ss_tag_frequency"]=json.dumps({"dataset":{trigger:1}})
    save_file(state,path,metadata=meta)

def pad_text(embeds,device):
    return ErnieImagePipeline._pad_text(embeds,device,torch.bfloat16,3072)

def sample_sigma(b):
    # Match the ERNIE FlowMatch scheduler's shift=4 schedule while sampling training noise uniformly.
    u=torch.rand((b,),device="cuda").clamp(1e-5,1-1e-5); shift=4.0
    return (shift*u/(1+(shift-1)*u)).to(torch.bfloat16)

class VAE:
    model=None
    @classmethod
    def get(cls):
        if cls.model is None: cls.model=AutoencoderKLFlux2.from_pretrained(MODEL,subfolder="vae",torch_dtype=torch.bfloat16).eval()
        return cls.model

def preview(model,scheduler,emb,shape,step):
    H,W=shape; model.eval(); device="cuda"
    text,lens=pad_text([emb],device)
    if PREVIEW_CFG>1:
        neg=torch.load(os.path.join(CACHE,"_neg_embed.pt"),weights_only=True)
        text,lens=pad_text([neg,emb],device)
    else: text,lens=pad_text([emb],device)
    gen=torch.Generator(device=device)
    gen.manual_seed((SEED if SEED>0 else int(time.time()))+step)
    lat=torch.randn((1,128,H,W),device=device,dtype=torch.bfloat16,generator=gen)
    sigmas=torch.linspace(1,0,PREVIEW_STEPS+1,device="cpu")[:-1]
    scheduler.set_timesteps(sigmas=sigmas,device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            model_lat=torch.cat([lat,lat]) if PREVIEW_CFG>1 else lat
            t_batch=torch.full((model_lat.shape[0],),float(t),device=device,dtype=torch.bfloat16)
            pred=model(hidden_states=model_lat,timestep=t_batch,text_bth=text,text_lens=lens,return_dict=False)[0]
            if PREVIEW_CFG>1:
                p0,p1=pred.chunk(2); pred=p0+PREVIEW_CFG*(p1-p0)
            lat=scheduler.step(pred,t,lat,return_dict=False)[0]
        vae=VAE.get().to(device)
        mean=vae.bn.running_mean.view(1,-1,1,1).to(device,lat.dtype); std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+1e-5).to(device,lat.dtype)
        z=ErnieImagePipeline._unpatchify_latents(lat[:1]*std+mean)
        img=vae.decode(z,return_dict=False)[0]
        arr=((img.float()/2+0.5).clamp(0,1)[0].cpu().permute(1,2,0).numpy()*255).astype("uint8")
        from PIL import Image
        Image.fromarray(arr).save(os.path.join(OUT,f"preview_step_{step}.png"))
        vae.to("cpu"); free()
    model.train()

def main():
    global RANK, ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for ERNIE training")
    if not os.path.isfile(os.path.join(CACHE,"_neg_embed.pt")): raise RuntimeError("Run ERNIE Pre-Cache first")
    model=load_nf4(MODEL).to("cuda"); model.enable_gradient_checkpointing()
    targets=[n for n,m in model.named_modules() if isinstance(m,(torch.nn.Linear,Linear4bit)) and (LORA_TARGETS=="all" or n.startswith("layers."))]
    if not targets: raise RuntimeError("No ERNIE transformer layer targets found for LoRA")
    print(f"Attaching {'all-linear' if LORA_TARGETS=='all' else 'block'} adapters (rank {RANK}) to {len(targets)} ERNIE linears",flush=True)
    resume_lokr=(os.path.isfile(os.path.join(RESUME,lora_options.LOKR_WEIGHTS))
                 and os.path.isfile(os.path.join(RESUME,lora_options.LOKR_CONFIG)))
    resume_lora=(os.path.isfile(os.path.join(RESUME,"adapter_model.safetensors"))
                 and os.path.isfile(os.path.join(RESUME,"adapter_config.json")))
    if (os.path.isfile(os.path.join(RESUME,lora_options.LOKR_WEIGHTS)) and not resume_lokr
            or os.path.isfile(os.path.join(RESUME,"adapter_model.safetensors")) and not resume_lora):
        print("[!] Incomplete ERNIE resume checkpoint detected; ignoring it and starting a new run.",flush=True)
    if resume_lokr:
        rc=json.load(open(os.path.join(RESUME,lora_options.LOKR_CONFIG),encoding="utf-8")); RANK=int(rc["r"]); ALPHA=float(rc["lora_alpha"]); LOKR_FACTOR=int(rc["lokr_factor"]); USE_LOKR=True
        with safe_open(os.path.join(RESUME,lora_options.LOKR_WEIGHTS),framework="pt",device="cpu") as f: targets=sorted({k.rpartition(".")[0] for k in f.keys() if k.endswith(".lokr_w1")})
    elif resume_lora:
        ac=json.load(open(os.path.join(RESUME,"adapter_config.json"),encoding="utf-8")); RANK=int(ac["r"]); ALPHA=float(ac["lora_alpha"]); USE_RSLORA=bool(ac.get("use_rslora",False))
        with safe_open(os.path.join(RESUME,"adapter_model.safetensors"),framework="pt",device="cpu") as f: targets=sorted({k.split(".lora_")[0].replace("base_model.model.","",1) for k in f.keys()})
        opts=os.path.join(RESUME,"lora_options.json")
        if os.path.isfile(opts):
            o=json.load(open(opts,encoding="utf-8")); USE_LORAPLUS=bool(o.get("use_loraplus",False)); LORAPLUS_RATIO=float(o.get("loraplus_ratio",LORAPLUS_RATIO))
    if USE_LOKR:
        USE_RSLORA=USE_LORAPLUS=False; model=lora_options.LoKrModel(model,targets,RANK,ALPHA,LOKR_FACTOR)
    else:
        lconf=LoraConfig(r=RANK,lora_alpha=ALPHA,target_modules=targets,bias="none",init_lora_weights=True,use_rslora=USE_RSLORA)
        model=get_peft_model(model,lconf)
    model.train()
    trainable=[p for p in model.parameters() if p.requires_grad]
    if USE_LORAPLUS:
        a=[p for n,p in model.named_parameters() if p.requires_grad and "lora_B" not in n]; b=[p for n,p in model.named_parameters() if p.requires_grad and "lora_B" in n]
        optimizer=torch.optim.AdamW([{"params":a,"lr_mult":1.0},{"params":b,"lr_mult":LORAPLUS_RATIO}],lr=LR,weight_decay=WEIGHT_DECAY)
    else: optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=WEIGHT_DECAY)
    for p in trainable: p.data=p.data.float()
    def lr_at(s):
        if s<WARMUP: return LR*s/max(WARMUP,1)
        q=(s-WARMUP)/max(1,STEPS-WARMUP); return LR*(MIN_LR_RATIO+(1-MIN_LR_RATIO)*.5*(1+math.cos(math.pi*min(1,q))))
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder="scheduler")
    cache={}; buckets=defaultdict(list)
    for f in os.listdir(CACHE):
        if not f.endswith("_latent.pt"): continue
        name=f[:-10]; lat=torch.load(os.path.join(CACHE,f),weights_only=True).to(torch.bfloat16).pin_memory(); emb=torch.load(os.path.join(CACHE,name+"_embed.pt"),weights_only=True).to(torch.bfloat16).pin_memory()
        cache[name]=(lat,emb); buckets[tuple(lat.shape[-2:])].append(name)
    if not cache: raise RuntimeError("No pre-cached ERNIE image latents found")
    step=0
    lora_path=os.path.join(RESUME,lora_options.LOKR_WEIGHTS if USE_LOKR else "adapter_model.safetensors")
    checkpoint_valid=(resume_lokr if USE_LOKR else resume_lora)
    if os.path.isfile(STEPFILE) and os.path.isfile(OPT) and checkpoint_valid and os.path.isfile(lora_path):
        try:
            step=int(open(STEPFILE,encoding="utf-8").read().strip()); sd=load_file(lora_path)
            if USE_LOKR: model.load_lokr_state_dict(sd)
            else: set_peft_model_state_dict(model,{k:v.float() for k,v in sd.items()})
            optimizer.load_state_dict(torch.load(OPT,weights_only=False)); print(f"Resuming at step {step}",flush=True)
            grad_path=os.path.join(RESUME,"gradients.pt")
            if os.path.isfile(grad_path):
                grads=torch.load(grad_path,weights_only=False)
                for n,p in model.named_parameters():
                    if n in grads and p.requires_grad: p.grad=grads[n].to(device=p.device,dtype=p.dtype)
        except Exception as e: print(f"Resume state ignored: {e}",flush=True); step=0
    def save():
        os.makedirs(RESUME,exist_ok=True); model.save_pretrained(RESUME)
        stale=os.path.join(RESUME,"adapter_model.safetensors" if USE_LOKR else lora_options.LOKR_WEIGHTS)
        if os.path.isfile(stale): os.remove(stale)
        with open(os.path.join(RESUME,"lora_options.json"),"w",encoding="utf-8") as f: json.dump({"use_loraplus":USE_LORAPLUS,"loraplus_ratio":LORAPLUS_RATIO},f)
        torch.save(optimizer.state_dict(),OPT); open(STEPFILE,"w",encoding="utf-8").write(str(step)); export(model,os.path.join(OUT,f"Ernie_LoRA_step_{step}.safetensors"),step); print(f"\nSaved ERNIE checkpoint at step {step}",flush=True)
        torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},os.path.join(RESUME,"gradients.pt"))
    def stop(sig,frame): save(); raise SystemExit(0)
    for sig in (signal.SIGINT,signal.SIGTERM,getattr(signal,"SIGBREAK",signal.SIGTERM)):
        try: signal.signal(sig,stop)
        except Exception: pass
    ordered=list(cache)
    print(f"ERNIE text-to-image training: {len(ordered)} images, {len(buckets)} buckets, {STEPS} steps",flush=True)
    model.train()
    running_loss=t_step_avg=grad_norm=0.0; start_step=step; last_step=step; live_mtime=None
    while step<STEPS:
        # Read GUI-saved settings at each step. Structural options stay fixed until resume.
        try:
            mt=os.path.getmtime(CONFIG)
            if live_mtime is None: live_mtime=mt
            elif mt!=live_mtime:
                fresh=json.load(open(CONFIG,encoding="utf-8")); live_mtime=mt; changed=[]
                for key,var,cast in (("total_steps","STEPS",int),("save_every","SAVE_EVERY",int),("lr","LR",float),("max_grad_norm","MAX_GRAD_NORM",float),("preview_every","PREVIEW_EVERY",int),("preview_steps","PREVIEW_STEPS",int),("preview_cfg","PREVIEW_CFG",float),("preview_size","PREVIEW_SIZE",int),("preview_caption_mode","PREVIEW_MODE",str),("preview_custom_prompt","CUSTOM",str),("seed","SEED",int)):
                    old=globals()[var]; new=cast(fresh.get(key,old)); globals()[var]=new
                    if old!=new: changed.append(f"{key}: {old} -> {new}")
                if changed: print("\n[LIVE] Settings reloaded: " + " | ".join(changed),flush=True)
        except (OSError,ValueError,TypeError): pass
        tic=time.time(); size=random.choice(list(buckets)); names=[random.choice(buckets[size]) for _ in range(BATCH)]
        z=torch.cat([cache[n][0] for n in names]).to("cuda",non_blocking=True); embeds=[cache[n][1] for n in names]
        text,lens=pad_text(embeds,"cuda"); b=z.shape[0]; sigma=sample_sigma(b); noise=torch.randn_like(z); se=sigma.view(b,1,1,1)
        noisy=((1-se)*z+se*noise).to(torch.bfloat16); target=noise-z; timestep=(sigma*1000).to(torch.bfloat16)
        pred=model(hidden_states=noisy,timestep=timestep,text_bth=text,text_lens=lens,return_dict=False)[0]
        loss=F.mse_loss(pred.float(),target.float())/ACCUM; loss.backward()
        running_loss+=loss.item()*ACCUM
        step+=1; last_step=step
        if step%ACCUM==0 or step==STEPS:
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,MAX_GRAD_NORM).item()
            for g in optimizer.param_groups: g["lr"]=lr_at(step)*g.get("lr_mult",1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        elapsed=time.time()-tic; t_step_avg=elapsed if not t_step_avg else .1*elapsed+.9*t_step_avg
        pct=min(1.0,step/max(1,STEPS))
        if PROGRESS_FILLED == "█":
            # Mantiene los 20 segmentos de Klein pero muestra avance fraccional desde el paso 1.
            fraction_chars=("", "▏", "▎", "▍", "▌", "▋", "▊", "▉")
            eighths=min(160,max(1,math.ceil(pct*20*8))); full,part=divmod(eighths,8)
            bar=PROGRESS_FILLED*full+fraction_chars[part]+PROGRESS_EMPTY*(20-full-(1 if part else 0))
        else:
            full=min(20,max(1,int(math.ceil(pct*20))))
            bar=PROGRESS_FILLED*full+PROGRESS_EMPTY*(20-full)
        eta=max(0,STEPS-step)*t_step_avg; eta_s=f"{int(eta//3600):02d}:{int(eta%3600//60):02d}:{int(eta%60):02d}"
        print(f"\rStep {step:4d}/{STEPS} [{bar}] {pct*100:5.1f}% | Loss {running_loss/max(1,step-start_step):.4f} | gnorm {grad_norm:.3f} | lr {lr_at(step):.2e} | {t_step_avg:.2f}s/it | ETA {eta_s}",end="",flush=True)
        if SAVE_EVERY>0 and step%SAVE_EVERY==0: save()
        if PREVIEW_EVERY>0 and step%PREVIEW_EVERY==0:
            if PREVIEW_MODE=="random": name=random.choice(ordered)
            elif PREVIEW_MODE=="rotate4": name=ordered[(step//max(1,PREVIEW_EVERY))%min(4,len(ordered))]
            else: name=ordered[0]
            e=cache[name][1]
            custom_path=os.path.join(CACHE,"_custom_embed.pt")
            if PREVIEW_MODE=="custom" and os.path.isfile(custom_path): e=torch.load(custom_path,weights_only=True)
            h,w=size
            if PREVIEW_SIZE>0:
                scale=PREVIEW_SIZE/max(h,w); h=max(1,round(h*scale)); w=max(1,round(w*scale))
            preview(model,scheduler,e,(h,w),step)
    save(); export(model,os.path.join(OUT,"Ernie_FINAL_LoRA.safetensors"),step); print("Training completed.",flush=True)

if __name__=="__main__": main()
