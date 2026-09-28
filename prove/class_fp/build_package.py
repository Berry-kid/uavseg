"""Build and verify a small, self-contained remote code bundle."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    existing=json.loads((HERE/"original_files_sha256.json").read_text(encoding="utf-8"))
    changes=[p for p,digest in existing.items() if sha(ROOT/p)!=digest]
    if changes:raise ValueError(f"Original inputs changed: {changes}")
    tests=json.loads((HERE/"smoke_d81c6ab2cd/test_results.json").read_text(encoding="utf-8"))
    if not tests["passed"] or tests["tests_run"]!=9:raise ValueError("Training smoke tests missing")
    validation={"training_integration_tests":tests,"diagnostic_unit_tests":{"run":10,"passed":True,
                "reported_by":"independent implementation/review agent"},
                "real_train_mask_count":6297,"original_weights_exactly_reproduced":True,
                "comparison_selfcheck_delta":0.0,"original_files_sha256_unchanged":True,
                "formal_training_completed":False,"real_score_improvement_verified":False,
                "shell_launcher_note":"Linux bash launcher supplied; local Windows host has no bash for execution."}
    (HERE/"validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    files=[ROOT/name for name in ("train.py","predict.py","select_tta.py","validate_submission.py","requirements.txt")]
    files+=sorted((ROOT/"uavseg").glob("*.py"))
    files+=[ROOT/"prove/training_monitor"/name for name in ("monitoring.py","train_monitored.py")]
    files+=[HERE/name for name in ("train_fp.py","weights.py","diagnostics.py","prepare_experiment.py",
                                  "compare_runs.py","test_diagnostics.py","test_training.py",
                                  "run_ablation.sh","README.md","validation.json")]
    files+=[HERE/"reference"/name for name in ("split.json","baseline.json","weight_options.json","BASELINE.md")]
    manifest={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in files}
    manifest_path=HERE/"package_manifest.json"
    manifest_path.write_text(json.dumps({"files_sha256":manifest,"contains_data":False,
        "contains_model_weights":False,"original_inputs_unchanged":True},indent=2),encoding="utf-8")
    files.append(manifest_path)
    archive=HERE/"qw-advice-class-fp.zip"
    if archive.exists():raise ValueError("Archive already exists")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in files:
            name=file.relative_to(ROOT).as_posix()
            z.write(file,name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError("ZIP CRC check failed")
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError("Duplicate archive paths")
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError("ZIP contents differ")
        shell=z.read("prove/class_fp/run_ablation.sh")
        assert shell.startswith(b"#!/usr/bin/env bash\n") and b"\r" not in shell
        dest=HERE/"package_smoke"
        if dest.exists():raise ValueError("Extraction smoke directory exists")
        dest.mkdir()
        for member in z.infolist():
            target=(dest/member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError("Unsafe archive path")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(member))
    result={"archive":str(archive),"bytes":archive.stat().st_size,"sha256":sha(archive),
            "files":len(files),"zip_integrity_and_source_hashes_verified":True,
            "original_inputs_unchanged":True,"extracted_smoke_directory":str(dest)}
    (HERE/"package_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":main()

