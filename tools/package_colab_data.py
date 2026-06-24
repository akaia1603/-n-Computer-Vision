"""
tools/package_colab_data.py
============================
Zip all files needed for Colab fine-tuning into a single package.
Run BEFORE uploading to Google Drive.
"""
import os
import sys
import zipfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT = "colab_upload"
os.system(f'rd /s /q "{OUTPUT}" 2>nul')
os.makedirs(OUTPUT, exist_ok=True)

# 1. Copy webcam zip
if os.path.exists("webcam_finetune_data.zip"):
    shutil.copy("webcam_finetune_data.zip", f"{OUTPUT}/webcam_finetune_data.zip")
    print("[OK] webcam_finetune_data.zip")
else:
    print("[SKIP] webcam_finetune_data.zip — run tools/prepare_finetune_data.py first")

# 2. Copy model files
for f in ["outputs/models/best_dan_model.pth", "outputs/models/poster_best.pth", "pretrain/ir50.pth", "pretrain/mobilefacenet_model_best.pth.tar"]:
    if os.path.exists(f):
        dst = f.replace("outputs/models/", "").replace("pretrain/", "")
        shutil.copy(f, f"{OUTPUT}/{dst}")
        print(f"[OK] {f}")
    else:
        print(f"[SKIP] {f} not found")

# 3. Copy models/ dir + app/models/poster_model.py (needed by Colab notebook)
models_dst = f"{OUTPUT}/models"
os.system(f'rd /s /q "{models_dst}" 2>nul')
if os.path.exists("models"):
    shutil.copytree("models", models_dst, ignore=shutil.ignore_patterns("__pycache__"))
if os.path.exists("app/models/poster_model.py"):
    shutil.copy("app/models/poster_model.py", f"{models_dst}/poster_model.py")
if os.path.exists("app/models/dan_model.py"):
    # Copy DAN's model definition too if it imports from root models
    pass  # DAN defines model inline in Colab notebook, no need
# Fix import name: hyp_crossvit -> hyvittransformer (notebook expects this name)
if os.path.exists(f"{models_dst}/hyp_crossvit.py") and not os.path.exists(f"{models_dst}/hyvittransformer.py"):
    shutil.copy(f"{models_dst}/hyp_crossvit.py", f"{models_dst}/hyvittransformer.py")
print("[OK] models/ (POSTER model code)")

# 4. Zip RAF-DB (data/DATASET)
rafdb_zip = f"{OUTPUT}/RAF-DB.zip"
if os.path.exists("data/DATASET/train") and os.path.exists("data/DATASET/test"):
    print("Zipping RAF-DB...")
    with zipfile.ZipFile(rafdb_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("data/DATASET"):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.relpath(full, "data/DATASET")
                zf.write(full, arcname)
    size_mb = os.path.getsize(rafdb_zip) / 1024 / 1024
    print(f"[OK] RAF-DB.zip ({size_mb:.0f} MB)")
else:
    print("[SKIP] data/DATASET/train or test missing")

print(f"\nAll files in '{OUTPUT}/':")
for fn in sorted(os.listdir(OUTPUT)):
    sz = os.path.getsize(os.path.join(OUTPUT, fn)) / 1024 / 1024
    print(f"  {fn:40s} {sz:.1f} MB")
print(f"\nUpload the entire '{OUTPUT}/' folder to Google Drive folder 'fer_finetune'.")
