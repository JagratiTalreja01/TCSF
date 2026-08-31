#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, importlib, inspect, json, logging, math, statistics, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets.dataloader import build_dataloader

MODEL_SPECS=[
{"name":"U-Net","factory":"test_unet:build_model","config":"configs/unet_base.yaml","checkpoint":"outputs/checkpoints/UNet_200ep_seed2026/best_model.pth"},
{"name":"DeepLabV3+","factory":"test_deeplabv3plus:build_model","config":"configs/deeplabv3plus_base.yaml","checkpoint":"outputs/checkpoints/DeepLabV3Plus_200ep_seed2026/best_model.pth"},
{"name":"Swin-UNet","factory":"test_swin_unet:build_model","config":"configs/swin_unet_base.yaml","checkpoint":"outputs/checkpoints/SwinUNet_200ep_seed2026/best_model.pth"},
{"name":"SegFormer-B0","factory":"test_segformer:build_model","config":"configs/segformer_base.yaml","checkpoint":"outputs/checkpoints/SegFormer_B0_200ep_seed2026/best_model.pth"},
{"name":"Vision Mamba","factory":"test_vision_mamba:build_model","config":"configs/vision_mamba_base.yaml","checkpoint":"outputs/checkpoints/VisionMamba_200ep_seed2026/best_model.pth"},
{"name":"TCSF v3.1","factory":"test:build_model","config":"configs/acsf_base.yaml","checkpoint":"outputs/checkpoints/TCSF_v31_final_200ep/best_model.pth"},
]

@dataclass
class Result:
    model:str; status:str="pending"; total_params_m:float=math.nan; trainable_params_m:float=math.nan; macs_g:float=math.nan; flops_g:float=math.nan; peak_memory_allocated_gb:float=math.nan; peak_memory_reserved_gb:float=math.nan; latency_mean_ms:float=math.nan; latency_median_ms:float=math.nan; latency_p95_ms:float=math.nan; fps:float=math.nan; test_time_s:float=math.nan; test_images:int=0; test_throughput_fps:float=math.nan; iou:float=math.nan; dice:float=math.nan; precision:float=math.nan; recall:float=math.nan; f1:float=math.nan; pixel_accuracy:float=math.nan; error:str=""

def setup_logger(out:Path):
    out.mkdir(parents=True,exist_ok=True); lg=logging.getLogger("benchmark"); lg.setLevel(logging.INFO); lg.handlers.clear(); fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"); sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh); fh=logging.FileHandler(out/"benchmark.log",mode="w"); fh.setFormatter(fmt); lg.addHandler(fh); return lg

def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f: d=yaml.safe_load(f)
    return d if isinstance(d,dict) else {}

def import_attr(path):
    m,a=path.rsplit(":",1); return getattr(importlib.import_module(m),a)

def build_model(spec,cfg,lg):
    f=import_attr(spec["factory"]); m=f(cfg); lg.info("Built %s via %s",spec["name"],spec["factory"]); return m

def extract_state_dict(c):
    if isinstance(c,Mapping):
        for k in ("model_state_dict","state_dict","model","net"):
            if isinstance(c.get(k),Mapping): return c[k]
        if c and all(torch.is_tensor(v) for v in c.values()): return c
    raise TypeError("Could not find state_dict")

def load_checkpoint(m,p,lg):
    c=torch.load(p,map_location="cpu"); s=extract_state_dict(c); clean={}
    for k,v in s.items():
        for pref in ("module.","model.","net."):
            if k.startswith(pref): k=k[len(pref):]
        clean[k]=v
    miss,unexp=m.load_state_dict(clean,strict=False); lg.info("Loaded %s | missing=%d unexpected=%d",p,len(miss),len(unexp))

def nested_get(d,paths,default):
    for path in paths:
        v=d; ok=True
        for k in path:
            if not isinstance(v,Mapping) or k not in v: ok=False; break
            v=v[k]
        if ok:return v
    return default

def make_inputs(cfg,dev,size):
    sc=int(nested_get(cfg,[("model","sar_channels"),("data","sar_channels")],2)); oc=int(nested_get(cfg,[("model","optical_channels"),("data","optical_channels")],13)); return torch.randn(1,sc,size,size,device=dev),torch.randn(1,oc,size,size,device=dev)

def forward_model(m,sar,opt):
    try:return m(sar,opt)
    except TypeError:return m(torch.cat([sar,opt],1))

def extract_logits(o):
    if torch.is_tensor(o): return o
    if isinstance(o,Mapping):
        for k in ("final_pred","logits","pred","prediction","out","fused_pred","mask"):
            if torch.is_tensor(o.get(k)): return o[k]
        for v in o.values():
            if torch.is_tensor(v) and v.ndim>=3:return v
    if isinstance(o,(tuple,list)):
        for v in o:
            try:return extract_logits(v)
            except:pass
    raise TypeError(type(o))

class Wrapper(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,sar,opt): return extract_logits(forward_model(self.m,sar,opt))

def count_flops(w,inputs,lg):
    try:
        from fvcore.nn import FlopCountAnalysis
        a=FlopCountAnalysis(w,inputs); a.unsupported_ops_warnings(False); a.uncalled_modules_warnings(False); macs=float(a.total()); return macs,macs
    except Exception as e: lg.warning("fvcore failed: %s",e)
    try:
        from thop import profile
        macs,_=profile(w,inputs=inputs,verbose=False); return float(macs),float(macs)
    except Exception as e: lg.warning("thop failed: %s",e)
    return math.nan,math.nan

@torch.no_grad()
def bench_latency(w,inputs,dev,warmup,iters,amp):
    for _ in range(warmup):
        with torch.cuda.amp.autocast(enabled=amp and dev.type=="cuda"): w(*inputs)
    if dev.type=="cuda": torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ts=[]
    for _ in range(iters):
        if dev.type=="cuda":
            s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record();
            with torch.cuda.amp.autocast(enabled=amp): w(*inputs)
            e.record(); torch.cuda.synchronize(); ts.append(float(s.elapsed_time(e)))
        else:
            t=time.perf_counter(); w(*inputs); ts.append((time.perf_counter()-t)*1000)
    mean=float(statistics.mean(ts)); out={"latency_mean_ms":mean,"latency_median_ms":float(statistics.median(ts)),"latency_p95_ms":float(np.percentile(np.asarray(ts),95)),"fps":1000/mean,"peak_memory_allocated_gb":math.nan,"peak_memory_reserved_gb":math.nan}
    if dev.type=="cuda": out["peak_memory_allocated_gb"]=torch.cuda.max_memory_allocated()/1024**3; out["peak_memory_reserved_gb"]=torch.cuda.max_memory_reserved()/1024**3
    return out

def parse_batch(b,dev):
    if isinstance(b,Mapping):
        sar=next((b[k] for k in ("sar","s1","sentinel1") if k in b),None); opt=next((b[k] for k in ("optical","s2","sentinel2","image") if k in b),None); mask=next((b[k] for k in ("mask","label","target","flood_mask") if k in b),None)
    else: sar,opt,mask=b[:3]
    sar=sar.float().to(dev); opt=opt.float().to(dev); mask=mask.float().to(dev); mask=mask.unsqueeze(1) if mask.ndim==3 else mask; return sar,opt,mask

def call_flexible(f,vals):
    sig=inspect.signature(f); kw={k:v for k,v in vals.items() if k in sig.parameters}; return f(**kw)

def discover_loader(cfg,config_path,bs,nw,lg):
    mods=["datasets.dataloader","datasets.sen1floods11","test","test_unet"]
    names=["build_test_loader","create_test_loader","get_test_loader","make_test_loader","build_dataloaders","create_dataloaders","get_dataloaders","build_loaders","get_loaders"]
    vals={"cfg":cfg,"config":cfg,"config_path":config_path,"batch_size":bs,"num_workers":nw,"split":"test","shuffle":False}
    for mn in mods:
        try:m=importlib.import_module(mn)
        except:continue
        for n in names:
            f=getattr(m,n,None)
            if not callable(f):continue
            try:r=call_flexible(f,vals)
            except Exception as e: lg.warning("%s.%s failed: %s",mn,n,e); continue
            if isinstance(r,Mapping):
                for k in ("test_loader","test","loader"):
                    if k in r:return r[k]
            if isinstance(r,(tuple,list)):
                return r[2] if len(r)>=3 else r[-1]
            if hasattr(r,"__iter__"):return r
    return None

class Metrics:
    def __init__(self):self.tp=self.fp=self.fn=self.tn=0
    def update(self,p,t):
        p=p.bool().reshape(-1); t=t.bool().reshape(-1); self.tp+=int((p&t).sum()); self.fp+=int((p&~t).sum()); self.fn+=int((~p&t).sum()); self.tn+=int((~p&~t).sum())
    def compute(self):
        e=1e-8; tp,fp,fn,tn=self.tp,self.fp,self.fn,self.tn; pr=tp/(tp+fp+e); re=tp/(tp+fn+e); return {"iou":tp/(tp+fp+fn+e),"dice":2*tp/(2*tp+fp+fn+e),"precision":pr,"recall":re,"f1":2*pr*re/(pr+re+e),"pixel_accuracy":(tp+tn)/(tp+fp+fn+tn+e)}

@torch.no_grad()
def evaluate(m,loader,dev,thr,amp):
    met=Metrics(); count=0; torch.cuda.synchronize() if dev.type=="cuda" else None; start=time.perf_counter()
    for b in loader:
        sar,opt,t=parse_batch(b,dev)
        with torch.cuda.amp.autocast(enabled=amp and dev.type=="cuda"): logits=extract_logits(forward_model(m,sar,opt))
        if logits.ndim==3:logits=logits.unsqueeze(1)
        if logits.shape[-2:]!=t.shape[-2:]:logits=F.interpolate(logits,t.shape[-2:],mode="bilinear",align_corners=False)
        if logits.shape[1]>1:logits=logits[:,1:2]
        probs=logits if logits.min().item()>=0 and logits.max().item()<=1 else torch.sigmoid(logits); met.update(probs>=thr,t>=0.5); count+=int(t.shape[0])
    torch.cuda.synchronize() if dev.type=="cuda" else None; elapsed=time.perf_counter()-start; out=met.compute(); out.update({"test_time_s":elapsed,"test_images":count,"test_throughput_fps":count/elapsed}); return out

def save(results,out):
    rows=[asdict(r) for r in results]; df=pd.DataFrame(rows); df.to_csv(out/"efficiency_metrics.csv",index=False); json.dump(rows,open(out/"efficiency_metrics.json","w"),indent=2); return df

def plot(df,col,ylabel,name,out):
    d=df[["model",col]].dropna();
    if d.empty:return
    fig,ax=plt.subplots(figsize=(9,5)); bars=ax.bar(d.model,d[col]); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=25); ax.grid(axis="y",alpha=.25)
    for b,v in zip(bars,d[col]):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    fig.tight_layout(); fig.savefig(out/f"{name}.png",dpi=300,bbox_inches="tight"); fig.savefig(out/f"{name}.pdf",bbox_inches="tight"); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda"); p.add_argument("--image-size",type=int,default=256); p.add_argument("--warmup",type=int,default=50); p.add_argument("--iterations",type=int,default=200); p.add_argument("--test-batch-size",type=int,default=1); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--threshold",type=float,default=.5); p.add_argument("--amp",action="store_true"); p.add_argument("--skip-test",action="store_true"); p.add_argument("--continue-on-error",action="store_true"); p.add_argument("--output-dir",default="outputs/publication/benchmark"); a=p.parse_args(); out=Path(a.output_dir); lg=setup_logger(out); dev=torch.device(a.device); lg.info("Device=%s PyTorch=%s",dev,torch.__version__); results=[]
    for spec in MODEL_SPECS:
        r=Result(spec["name"])
        try:
            lg.info("="*70); lg.info("Benchmarking %s",spec["name"]); cfg=load_yaml(spec["config"]); m=build_model(spec,cfg,lg); load_checkpoint(m,spec["checkpoint"],lg); m=m.to(dev).eval(); r.total_params_m=sum(p.numel() for p in m.parameters())/1e6; r.trainable_params_m=sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6; inputs=make_inputs(cfg,dev,a.image_size); w=Wrapper(m).to(dev).eval(); macs,flops=count_flops(w,inputs,lg); r.macs_g=macs/1e9 if not math.isnan(macs) else math.nan; r.flops_g=flops/1e9 if not math.isnan(flops) else math.nan
            for k,v in bench_latency(w,inputs,dev,a.warmup,a.iterations,a.amp).items():setattr(r,k,v)
            if not a.skip_test:
                loader=build_dataloader(cfg,split="test")
                lg.info("Loaded test dataloader | batches=%d samples=%d",len(loader),len(loader.dataset))
                for k,v in evaluate(m,loader,dev,a.threshold,a.amp).items():
                    setattr(r,k,v)
            r.status="ok"; lg.info("%s | Params %.3fM | Complexity %s | Mem %.3fGB | Latency %.3fms | FPS %.2f | Test %.2fs | IoU %.4f | Dice %.4f",r.model,r.trainable_params_m,f"{r.flops_g:.3f}G" if not math.isnan(r.flops_g) else "N/A",r.peak_memory_allocated_gb,r.latency_mean_ms,r.fps,r.test_time_s,r.iou,r.dice); del m,w,inputs; gc.collect(); torch.cuda.empty_cache() if dev.type=="cuda" else None
        except Exception as e:
            lg.exception("Failed %s",spec["name"]); r.status="failed"; r.error=str(e)
            if not a.continue_on_error: results.append(r); save(results,out); raise
        results.append(r); df=save(results,out)
        for col,yl,n in [("trainable_params_m","Trainable parameters (M)","params_comparison"),("flops_g","FLOPs (G)","flops_comparison"),("peak_memory_allocated_gb","Peak GPU memory (GB)","memory_comparison"),("latency_mean_ms","Latency (ms/image)","latency_comparison"),("fps","FPS","fps_comparison")]:plot(df,col,yl,n,out)
    lg.info("Complete: %s",out)
if __name__=="__main__":main()