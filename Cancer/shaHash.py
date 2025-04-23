import hashlib

with open('cancer_pred.pkl', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
print(file_hash)