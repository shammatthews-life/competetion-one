import os
import hashlib

root = r"D:\gci\champion"
exclude_dirs = {".git", "clean_room_build", ".venv", "__pycache__", "catboost_info"}
exclude_files = {"MASTER_FILE_INVENTORY.csv", "SHA256SUMS.txt", "FINAL_SHA256_VERIFICATION.csv", "LOCAL_GITHUB_PARITY_REPORT.md"}

files_info = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
    for f in sorted(filenames):
        if f in exclude_files:
            continue
        full_path = os.path.join(dirpath, f)
        rel_path = os.path.relpath(full_path, root).replace("/", "\\")
        size = os.path.getsize(full_path)
        with open(full_path, "rb") as fp:
            sha = hashlib.sha256(fp.read()).hexdigest()
        files_info.append((rel_path, size, sha))

files_info.sort(key=lambda x: x[0])

with open(os.path.join(root, "MASTER_FILE_INVENTORY.csv"), "w", encoding="utf-8") as f_inv:
    f_inv.write("relative_path,size_bytes,sha256\n")
    for rel_path, size, sha in files_info:
        f_inv.write(f"{rel_path},{size},{sha}\n")

with open(os.path.join(root, "SHA256SUMS.txt"), "w", encoding="utf-8") as f_sums:
    for rel_path, size, sha in files_info:
        f_sums.write(f"{sha}  {rel_path}\n")

print(f"Cataloged {len(files_info)} files into MASTER_FILE_INVENTORY.csv and SHA256SUMS.txt.")
