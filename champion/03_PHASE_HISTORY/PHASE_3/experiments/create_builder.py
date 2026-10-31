import os

src_file = r'D:\gci\gemini\build_hasan_600_features.py'
dst_file = r'D:\gci\gemini\champion\phase3_reconstruction\experiments\run_hasan_600_builder.py'

with open(src_file, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace("train_path = '../inpuy/train.csv'", "train_path = r'D:/gci/inpuy/train.csv'")
code = code.replace("test_path = '../inpuy/test.csv'", "test_path = r'D:/gci/inpuy/test.csv'")
code = code.replace("os.makedirs('cache', exist_ok=True)", "os.makedirs(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild', exist_ok=True)")
code = code.replace("X_tr_hasan.to_feather('cache/hasan_features_tr.feather')", "X_tr_hasan.to_feather(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild/hasan_features_tr.feather')")
code = code.replace("X_te_hasan.to_feather('cache/hasan_features_te.feather')", "X_te_hasan.to_feather(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild/hasan_features_te.feather')")

with open(dst_file, 'w', encoding='utf-8') as f:
    f.write(code)

print("Created run_hasan_600_builder.py successfully.")

