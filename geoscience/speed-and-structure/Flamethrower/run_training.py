import subprocess
import yaml

# Load YAML fold assignments
with open("model_folds.yaml", "r") as f:
    assignments = yaml.safe_load(f)

for model_name, folds in assignments.items():
    script_name = f"src/train_{model_name}.py"
    
    for fold in folds:
        cmd = [
            "python",
            script_name,
            "--fold", str(fold),
            "--data_dir", "./data/",
            "--sub_dir", "./Submissions"
        ]
        
        print(f"\n>>> Running {model_name} | Fold {fold}")
        subprocess.run(cmd, check=True)
