import subprocess
import os

def resolve_merge_conflicts():
    """Resolve the merge conflicts and complete the push"""
    
    project_path = r"D:\Final Project\mlflow_project"
    os.chdir(project_path)
    
    print("🔧 Resolving merge conflicts...")
    
    try:
        # Check current conflict status
        print("📋 Current conflict status:")
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        print(result.stdout)
        
        # The conflict is about models.dvc file location
        # Let's check what files exist
        print("\n📁 Checking files:")
        if os.path.exists("mlflow_project/models/models.dvc"):
            print("✅ mlflow_project/models/models.dvc exists")
        if os.path.exists("models/models.dvc"):
            print("✅ models/models.dvc exists")
        
        # Since we want to keep the current structure (models/models.dvc),
        # let's accept our current version and remove the old path
        print("\n🔄 Resolving conflicts...")
        
        # Remove the old path if it exists
        if os.path.exists("mlflow_project/"):
            subprocess.run(["git", "rm", "-r", "mlflow_project/"], capture_output=True)
            print("✅ Removed old mlflow_project/ path")
        
        # Keep our current models/models.dvc
        subprocess.run(["git", "add", "models/models.dvc"], check=True)
        print("✅ Kept models/models.dvc")
        
        # Add any other files that might be in conflict
        subprocess.run(["git", "add", "."], check=True)
        
        # Check status after resolution
        print("\n📋 Status after resolution:")
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        print(result.stdout)
        
        # Commit the resolution
        print("💾 Committing conflict resolution...")
        subprocess.run(["git", "commit", "-m", "Resolve merge conflicts: keep current models structure"], check=True)
        
        # Configure DVC remote if not done
        print("📡 Configuring DVC remote...")
        subprocess.run(["dvc", "remote", "add", "origin", "https://dagshub.com/Mostafa2074/my-first-repo.dvc"], capture_output=True)
        subprocess.run(["dvc", "remote", "default", "origin"], check=True)
        
        # Push to DagsHub
        print("🚀 Pushing to DagsHub...")
        subprocess.run(["git", "push", "origin", "master"], check=True)
        
        # Push to DVC remote
        print("📦 Pushing models to DVC...")
        subprocess.run(["dvc", "push"], check=True)
        
        print("\n✅ SUCCESS: Conflicts resolved and everything pushed to DagsHub!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Error output: {e.stderr}")

# Run the resolution
resolve_merge_conflicts()