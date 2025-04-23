import hashlib
for file in ['./Heart Disease/heart_failure_model.pkl', './Heart Disease/scaler.pkl']:
    with open(file, 'rb') as f:
        print(f"{file}: {hashlib.sha256(f.read()).hexdigest()}")