"""
HuggingFace Spaces Deployment Script
=====================================
Automatically packages your trained models and pushes to HF Spaces.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   (uses your HF token)

Usage:
    python deploy_to_hf.py --username YOUR_HF_USERNAME --space-name scam-detection-api
"""

import os
import shutil
import argparse
from pathlib import Path


def deploy(username: str, space_name: str):
    """Package and deploy to HuggingFace Spaces."""
    from huggingface_hub import HfApi, create_repo, upload_folder

    repo_id = f"{username}/{space_name}"
    api = HfApi()

    # 1. Create repo
    print(f"🚀 Creating HuggingFace Space: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"   ✅ Space ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"   ⚠️  Repo creation: {e}")

    # 2. Build deploy directory
    deploy_dir = Path("hf_deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()

    # Copy project structure
    structure = {
        "url_model": ["url_feature_extractor.py"],
        "message_model": ["train_message_model.py"],
        "api": ["app.py"],
        "models": [],  # trained model files
        ".": ["requirements.txt"],
    }

    for src_dir, files in structure.items():
        src_path = Path(src_dir) if src_dir != "." else Path(".")
        dest_path = deploy_dir / src_dir if src_dir != "." else deploy_dir

        if files:
            dest_path.mkdir(exist_ok=True, parents=True)
            for f in files:
                src_file = src_path / f
                if src_file.exists():
                    shutil.copy2(src_file, dest_path / f)
                    print(f"   📁 Copied {src_file}")
                else:
                    print(f"   ⚠️  Missing: {src_file}")

    # Copy entire models directory (trained weights)
    models_src = Path("models")
    models_dest = deploy_dir / "models"
    if models_src.exists():
        shutil.copytree(models_src, models_dest)
        print(f"   📁 Copied models/ directory")
    else:
        print("   ❌ models/ directory not found! Train models first.")
        models_dest.mkdir(exist_ok=True)

    # Copy Dockerfile and README
    shutil.copy2("deployment/Dockerfile", deploy_dir / "Dockerfile")
    shutil.copy2("deployment/README.md", deploy_dir / "README.md")

    # Add __init__.py files
    for d in ["url_model", "message_model", "api"]:
        (deploy_dir / d / "__init__.py").touch()

    # 3. Upload to HF
    print(f"\n📤 Uploading to HuggingFace Spaces...")
    api.upload_folder(
        folder_path=str(deploy_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["*.pyc", "__pycache__"],
    )

    print(f"\n✅ Deployment complete!")
    print(f"   Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"   API URL:   https://{username.lower()}-{space_name.lower()}.hf.space")
    print(f"\n   Test your API:")
    print(f"   curl -X POST https://{username.lower()}-{space_name.lower()}.hf.space/detect \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\"message\": \"URGENT: Verify account!\", \"url\": \"http://paypal-secure.tk\"}}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument("--space-name", default="scam-detection-api",
                        help="Space name (default: scam-detection-api)")
    args = parser.parse_args()
    deploy(args.username, args.space_name)
