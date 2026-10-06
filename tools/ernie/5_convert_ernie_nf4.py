# -*- coding: utf-8 -*-
"""Convert ERNIE-Image Diffusers transformer and Mistral3 text encoder to NF4."""
import gc, json, os, shutil
import torch
from safetensors.torch import save_file

RAW = "Ernie-Raw"
OUT = "Ernie-NF4"
DTYPE = torch.bfloat16

def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def quantize_transformer():
    from diffusers import ErnieImageTransformer2DModel
    from bitsandbytes.nn import Linear4bit, Params4bit
    print("Loading ERNIE transformer on CPU; quantization runs one layer at a time to fit 16 GB GPUs.", flush=True)
    model = ErnieImageTransformer2DModel.from_pretrained(RAW, subfolder="transformer", torch_dtype=DTYPE, device_map="cpu")
    root = os.path.join(OUT, "transformer")
    weights = os.path.join(root, "weights")
    os.makedirs(weights, exist_ok=True)
    index = {"quantized": {}, "unquantized": {}, "others": "others.safetensors"}
    saved = set()
    # ERNIE's 36 shared AdaLN blocks hold essentially the whole DiT; peripheral projections stay BF16.
    for name, module in list(model.named_modules()):
        if not name.startswith("layers.") or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        src = getattr(parent, child_name)
        print("NF4", name, flush=True)
        layer = Linear4bit(src.in_features, src.out_features, bias=src.bias is not None,
                           quant_type="nf4", compute_dtype=DTYPE)
        layer.weight = Params4bit(src.weight.detach().float().contiguous(), requires_grad=False, quant_type="nf4")
        if src.bias is not None:
            layer.bias = torch.nn.Parameter(src.bias.detach().clone(), requires_grad=False)
        layer.to("cuda")
        qweight = layer.weight
        if not getattr(qweight, "bnb_quantized", False) or qweight.quant_state is None:
            raise RuntimeError(f"NF4 quantization failed for {name}")
        tensors = {"weight": qweight.data.detach().cpu().contiguous()}
        if layer.bias is not None: tensors["bias"] = layer.bias.detach().to(DTYPE).cpu().contiguous()
        qs = qweight.quant_state.as_dict(packed=True)
        for k,v in qs.items():
            if torch.is_tensor(v): tensors["quant_state."+k] = v.detach().cpu().contiguous()
        fname = name.replace(".", "__") + ".safetensors"
        save_file(tensors, os.path.join(weights, fname), metadata={"layer_name":name,"type":"Linear4bit","quant_type":"nf4","in_features":str(src.in_features),"out_features":str(src.out_features),"bias":str(src.bias is not None)})
        index["quantized"][name] = {"file":fname,"in_features":src.in_features,"out_features":src.out_features,"bias":src.bias is not None,"quant_type":"nf4","compute_dtype":"bfloat16","quant_state_keys":list(qs)}
        saved.update((name+".weight", name+".bias"))
        setattr(parent, child_name, torch.nn.Identity())
        del layer, qweight, tensors, qs
        free()
    others = {n:p.detach().to(DTYPE).cpu().contiguous() for n,p in list(model.named_parameters())+list(model.named_buffers()) if n not in saved}
    save_file(others, os.path.join(root,"others.safetensors"))
    with open(os.path.join(root,"index.json"),"w",encoding="utf-8") as f: json.dump(index,f,indent=2)
    shutil.copy2(os.path.join(RAW,"transformer","config.json"),os.path.join(root,"config.json"))
    with open(os.path.join(root,"metadata.json"),"w",encoding="utf-8") as f:
        json.dump({"format":"ErnieImage-NF4","version":1,"model_id":"baidu/ERNIE-Image","dtype":"bfloat16","quant_type":"nf4","quantized_layers":len(index["quantized"]),"quant_blocks":["layers"],"loader":"Params4bit.from_prequantized"},f,indent=2)
    print(f"Transformer NF4 saved: {len(index['quantized'])} layers, {len(others)} other tensors.",flush=True)
    del model, others; free()

def quantize_text_encoder():
    from transformers import BitsAndBytesConfig, Mistral3Model
    print("Loading Mistral3 text encoder in NF4...", flush=True)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=DTYPE,bnb_4bit_use_double_quant=True)
    te=Mistral3Model.from_pretrained(os.path.join(RAW,"text_encoder"),quantization_config=cfg,torch_dtype=DTYPE,device_map={"":"cuda"})
    te.save_pretrained(os.path.join(OUT,"text_encoder"),safe_serialization=True)
    del te; free()

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for NF4 conversion.")
    if not os.path.isfile(os.path.join(RAW,"transformer","config.json")): raise FileNotFoundError("ERNIE Diffusers raw model is missing. Download baidu/ERNIE-Image into Ernie-Raw first.")
    os.makedirs(OUT,exist_ok=True)
    # Keep the Diffusers pipeline's non-weight components; PE (prompt enhancer) is not used for training.
    for name in os.listdir(RAW):
        if name in ("transformer","text_encoder","pe","pe_tokenizer",".cache"): continue
        src=os.path.join(RAW,name); dst=os.path.join(OUT,name)
        if os.path.isdir(src): shutil.copytree(src,dst,dirs_exist_ok=True)
        elif os.path.isfile(src): shutil.copy2(src,dst)
    if os.path.isdir(os.path.join(OUT,"transformer")): shutil.rmtree(os.path.join(OUT,"transformer"))
    quantize_transformer()
    quantize_text_encoder()
    print("ERNIE-Image NF4 conversion complete:",os.path.abspath(OUT),flush=True)

if __name__=="__main__": main()
