import os
import subprocess

def fix_unclickable_mlflow():
    """Fix the unclickable mlflow_project on DagsHub"""
    
    base_dir = r"D:\Final Project"
    os.chdir(base_dir)
    
    print("🔍 Diagnosing unclickable mlflow_project...")
    
    mlflow_path = "mlflow_project"
    
    if not os.path.exists(mlflow_path):
        print("❌ mlflow_project doesn't exist locally")
        return
    
    # Check what it is
    if os.path.isdir(mlflow_path):
        contents = os.listdir(mlflow_path)
        if not contents:
            print("📁 mlflow_project is an EMPTY folder")
            print("🔄 Removing empty folder...")
            subprocess.run(["git", "rm", "-r", mlflow_path], check=True)
        else:
            print(f"📁 mlflow_project is a folder with {len(contents)} items:")
            for item in contents:
                print(f"  - {item}")
            # Check if files are tracked
            result = subprocess.run(["git", "ls-files", mlflow_path], capture_output=True, text=True)
            if not result.stdout.strip():
                print("❌ Folder exists but no files are tracked by Git")
    
    elif os.path.isfile(mlflow_path):
        print("📄 mlflow_project is a FILE (not a folder)")
        print("🔄 Removing file...")
        subprocess.run(["git", "rm", mlflow_path], check=True)
    
    # Commit and push if we made changes
    result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "commit", "-m", "Fix unclickable mlflow_project"], check=True)
        subprocess.run(["git", "push", "origin", "master"], check=True)
        print("✅ Fixed and pushed to DagsHub!")
    else:
        print("ℹ️ No changes needed")

# Run the fix
fix_unclickable_mlflow()