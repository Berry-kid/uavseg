"""Compare final TTA results under the same saved validation protocol."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from prove.class_fp.diagnostics import summarize_confusion
from uavseg.selection import file_hash, write_json

LABELS=["背景","建筑","道路","水体","荒地","森林","农田","车辆"]

def compare(run, reference):
    baseline=json.loads(reference.read_text(encoding="utf-8"))
    selection=json.loads((run/"tta_selection.json").read_text(encoding="utf-8"))
    if not selection.get("complete") or selection["validation_images"]!=baseline["validation_images"]:
        raise ValueError("Incomplete or different validation set")
    if file_hash(run/"split.json")!=baseline["split_sha256"]:
        raise ValueError("Validation split differs from baseline")
    if selection["inference_config"]!=baseline["inference_config"]:
        raise ValueError("TTA settings differ from baseline")
    chosen=next(r for r in selection["candidate_results"] if r["epoch"]==selection["selected_epoch"])
    if not chosen.get("complete") or chosen["evaluated_images"]!=baseline["validation_images"]:
        raise ValueError("Selected candidate has not completed the complete validation set")
    before=np.array(baseline["confusion_matrix"],dtype=np.int64)
    after=np.array(chosen["confusion_matrix"],dtype=np.int64)
    if before.shape!=(9,9) or after.shape!=(9,9) or not np.array_equal(before.sum(1),after.sum(1)):
        raise ValueError("Ground-truth pixel counts differ; datasets may have changed")
    def miou(cm):
        tp=cm.diagonal()
        union=cm.sum(0)+cm.sum(1)-tp
        return float((tp[1:]/np.maximum(union[1:],1)).mean())
    if abs(miou(before)-baseline["tta_miou"])>1e-10 or abs(miou(after)-chosen["tta_miou"])>1e-10:
        raise ValueError("Recorded mIoU disagrees with the saved confusion matrix")
    a,b=summarize_confusion(before),summarize_confusion(after)
    rows=[]
    for old,new in zip(a["per_class"],b["per_class"]):
        row={"class_id":old["class_id"],"class_name":old["class_name"],"before":old,"after":new}
        row["delta"]={k:new[k]-old[k] if new[k] is not None and old[k] is not None else None
                      for k in ("iou","precision","recall","fp","fn")}
        rows.append(row)
    return {"complete":True,"baseline_tta_miou":baseline["tta_miou"],
            "candidate_tta_miou":chosen["tta_miou"],
            "delta_percentage_points":100*(chosen["tta_miou"]-baseline["tta_miou"]),
            "selected_epoch":chosen["epoch"],"validation_images":baseline["validation_images"],
            "same_split_same_tta_same_ground_truth_counts":True,"per_class":rows,
            "reference":str(reference.resolve()),"run":str(run.resolve()),
            "note":"Compare IoU together with precision/recall. Lower FP alone is not success; weighted CE losses across schemes are not comparable."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=Path(__file__).resolve().parent/"reference/baseline.json")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=compare(args.run.resolve(),args.reference.resolve())
    out=(args.output or args.run/"fp_comparison").resolve()
    if not out.is_relative_to(ROOT/"prove"):
        raise ValueError("Comparison output must stay within prove/")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite an existing comparison")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"comparison.json",result)
    f=lambda x:"未定义" if x is None else f"{x*100:.3f}%"
    lines=["# 类别误报实验对照","",
           f"完整 val + TTA mIoU：{result['baseline_tta_miou']:.6f} → {result['candidate_tta_miou']:.6f}，变化 {result['delta_percentage_points']:+.4f} 个百分点。","",
           "| 类别 | 原IoU | 新IoU | 原精确率 | 新精确率 | 原召回率 | 新召回率 | FP像素变化 | FN像素变化 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["per_class"]:
        a,b,d=row["before"],row["after"],row["delta"]
        lines.append(f"| {LABELS[row['class_id']-1]} | {f(a['iou'])} | {f(b['iou'])} | {f(a['precision'])} | {f(b['precision'])} | {f(a['recall'])} | {f(b['recall'])} | {d['fp']:+d} | {d['fn']:+d} |")
    lines+=["","划分SHA256、TTA配置及真实标签各类像素总数已核对一致。",
            "是否保留改动主要看完整val mIoU与类别IoU；误报减少但漏检大增不算改善。",
            "不同类别权重下的加权CE loss不能直接横向比较；本地改善也不保证线上同幅度改善。"]
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out/"COMPARISON.md")
    print(json.dumps({"baseline":result["baseline_tta_miou"],"candidate":result["candidate_tta_miou"],
                      "delta_pp":result["delta_percentage_points"]}))

if __name__=="__main__":main()

