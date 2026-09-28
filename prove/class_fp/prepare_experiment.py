"""Prepare train-only weight evidence and a portable fixed-split experiment reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from prove.class_fp.weights import count_train_pixels, weight_report
from uavseg.selection import file_hash, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--baseline-run",type=Path,default=Path("runs/runs/continue-v1"))
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"reference")
    a=p.parse_args()
    data,run,out=a.data_root.resolve(),a.baseline_run.resolve(),a.output.resolve()
    if not out.is_relative_to(ROOT/"prove/class_fp"):
        raise ValueError("Reference output must stay inside prove/class_fp")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite an existing reference")
    split_path=run/"split.json"
    split_bytes=split_path.read_bytes()
    split=json.loads(split_bytes)
    train,val=split["train"],split["val"]
    names=sorted(p.name for p in (data/"images").glob("*.png"))
    if (set(train)&set(val) or len(set(train))!=len(train) or len(set(val))!=len(val)
            or set(train)|set(val)!=set(names)):
        raise ValueError("Invalid saved split")
    info=json.loads((run/"monitor/run_info.json").read_text(encoding="utf-8"))
    selection=json.loads((run/"tta_selection.json").read_text(encoding="utf-8"))
    chosen=next(r for r in selection["candidate_results"] if r["epoch"]==selection["selected_epoch"])
    mask_paths=[data/"masks"/n for n in train]
    before={n:[p.stat().st_size,p.stat().st_mtime_ns] for n,p in zip(train,mask_paths)}
    counts=count_train_pixels(data,train,workers=4,
        progress=lambda n,total:print(f"Train-only mask count: {n}/{total}",flush=True))
    if before!={n:[p.stat().st_size,p.stat().st_mtime_ns] for n,p in zip(train,mask_paths)}:
        raise ValueError("Training masks changed during counting")
    from prove.class_fp.diagnostics import summarize_confusion
    modes={m:weight_report(counts,train,m) for m in ("inverse","inverse_sqrt","none")}
    legacy=info["class_weights"][1:]
    actual=modes["inverse"]["class_weights_ids_0_to_8"][1:]
    if legacy!=actual:
        raise ValueError(f"Recounted weights do not exactly match the baseline: {actual} versus {legacy}")
    out.mkdir(parents=True,exist_ok=True)
    (out/"split.json").write_bytes(split_bytes)
    diagnostic=summarize_confusion(chosen["confusion_matrix"])
    baseline={"source_run":str(run),"checkpoint_epoch":selection["selected_epoch"],
              "checkpoint_sha256":file_hash(run/"best_tta.pt"),
              "split_sha256":hashlib.sha256(split_bytes).hexdigest(),
              "validation_images":len(val),"training_images":len(train),
              "tta_miou":chosen["tta_miou"],"inference_config":selection["inference_config"],
              "training_args":info["args"],"confusion_matrix":chosen["confusion_matrix"],
              "false_positive_diagnostics":diagnostic,
              "online_score_user_reported":65.6259,"online_miou_user_reported":0.656259,
              "online_class_metrics_available":False}
    write_json(out/"baseline.json",baseline)
    write_json(out/"weight_options.json",modes)
    write_json(out/"training_mask_manifest.json",{"input_stats":before})
    labels=["背景","建筑","道路","水体","荒地","森林","农田","车辆"]
    lines=["# 类别误报实验依据","","仅从原训练划分的6297张mask计算权重；val/test像素未参与。","",
           "| ID | 类别 | 训练像素数 | 原倒数权重 | 平方根倒数权重 |",
           "|---|---|---:|---:|---:|"]
    w1=modes["inverse"]["class_weights_ids_0_to_8"]
    w2=modes["inverse_sqrt"]["class_weights_ids_0_to_8"]
    for i,label in enumerate(labels,1):
        lines.append(f"| {i} | {label} | {counts[i]} | {w1[i]:.6f} | {w2[i]:.6f} |")
    lines+=["",f"车辆/背景权重比：{w1[8]/w1[1]:.4f} → {w2[8]/w2[1]:.4f}。",
            f"荒地/背景权重比：{w1[5]/w1[1]:.4f} → {w2[5]/w2[1]:.4f}。",
            "",f"当前模型完整val+TTA mIoU：{baseline['tta_miou']:.8f}。",
            "分项误报、精确率和召回率见 baseline.json 中 false_positive_diagnostics。",
            "", "本次假设：缓和稀有类别权重可能减少过度预测。是否有效必须训练后验证，不能把权重数值变化当成分数提升。",
            "不同权重方案的加权CE loss不直接比较；主要比较同一完整val+TTA下的mIoU、各类IoU、精确率和召回率。"]
    (out/"BASELINE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"training_images":len(train),"validation_images":len(val),
                      "original_weights_exactly_reproduced":True,"weight_ratios":{
                          "vehicle_before":w1[8]/w1[1],"vehicle_after":w2[8]/w2[1],
                          "barren_before":w1[5]/w1[1],"barren_after":w2[5]/w2[1]}},ensure_ascii=False))

if __name__=="__main__":
    main()

