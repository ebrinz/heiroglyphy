#!/usr/bin/env python3
"""
Fix gensim/numpy compatibility for the spontaneous-remission kernel.
This script installs numpy<2.0 which is compatible with gensim 4.x.
"""

import subprocess
import sys

def main():
    print("Fixing gensim/numpy compatibility...")
    print("=" * 60)
    
    # Install compatible numpy version
    print("\n1. Installing numpy<2.0...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--quiet", "numpy<2.0"
    ])
    print("   ✓ Installed numpy<2.0")
    
    # Reinstall gensim to ensure it's compiled against the right numpy
    print("\n2. Reinstalling gensim...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "uninstall", "-y", "gensim"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--no-cache-dir", "gensim"
    ])
    print("   ✓ Reinstalled gensim")
    
    # Test the import
    print("\n3. Testing import...")
    try:
        from gensim.models import FastText
        print("   ✓ Gensim loaded successfully!")
        print("\n" + "=" * 60)
        print("SUCCESS! You can now use gensim in your notebook.")
        print("Please restart your Jupyter kernel to use the updated packages.")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
