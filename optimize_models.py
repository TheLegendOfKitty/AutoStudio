#!/usr/bin/env python3
"""
Model Storage Optimizer

Optimizes existing model downloads by:
1. Removing duplicate files
2. Converting split files to single files where possible
3. Cleaning up unnecessary cache
4. Providing storage analytics
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
import hashlib
from collections import defaultdict

class ModelOptimizer:
    """Optimize model storage and remove redundancies"""
    
    def __init__(self):
        self.duplicate_files = defaultdict(list)
        self.total_saved = 0
        
    def get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of a file"""
        if not file_path.exists() or not file_path.is_file():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def find_duplicate_files(self, search_dirs: List[str]) -> Dict[str, List[Path]]:
        """Find duplicate files across directories"""
        print("🔍 Scanning for duplicate files...")
        
        file_hashes = defaultdict(list)
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if not search_path.exists():
                continue
                
            print(f"   Scanning {search_path}...")
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    file_hash = self.get_file_hash(file_path)
                    if file_hash:
                        file_hashes[file_hash].append(file_path)
        
        # Filter to only duplicates
        duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
        
        print(f"📊 Found {len(duplicates)} sets of duplicate files")
        return duplicates
    
    def remove_duplicates(self, duplicates: Dict[str, List[Path]], dry_run: bool = True) -> float:
        """Remove duplicate files, keeping the one in the best location"""
        total_saved = 0
        
        for file_hash, file_paths in duplicates.items():
            if len(file_paths) <= 1:
                continue
            
            # Sort by preference: prefer models_efficient > models > cache
            def path_priority(path: Path) -> int:
                path_str = str(path)
                if "models_efficient" in path_str:
                    return 0  # Highest priority
                elif "models" in path_str and "cache" not in path_str:
                    return 1  # Medium priority  
                elif "cache" in path_str:
                    return 2  # Lowest priority
                else:
                    return 3  # Unknown
            
            sorted_paths = sorted(file_paths, key=path_priority)
            keep_file = sorted_paths[0]
            remove_files = sorted_paths[1:]
            
            print(f"\n📄 Duplicate set ({len(file_paths)} files):")
            print(f"   Keep: {keep_file}")
            
            for remove_file in remove_files:
                file_size = remove_file.stat().st_size
                total_saved += file_size
                
                print(f"   {'[DRY RUN] ' if dry_run else ''}Remove: {remove_file} ({file_size / 1024**2:.1f} MB)")
                
                if not dry_run:
                    try:
                        remove_file.unlink()
                    except Exception as e:
                        print(f"   ❌ Failed to remove: {e}")
        
        return total_saved / (1024**3)  # Convert to GB
    
    def analyze_model_storage(self, base_dirs: List[str]):
        """Analyze and report on model storage usage"""
        print("\n📊 Model Storage Analysis")
        print("=" * 50)
        
        total_size = 0
        dir_sizes = {}
        
        for base_dir in base_dirs:
            base_path = Path(base_dir)
            if not base_path.exists():
                continue
                
            dir_size = self.get_directory_size(base_path)
            dir_sizes[base_dir] = dir_size
            total_size += dir_size
            
            print(f"📁 {base_dir}: {dir_size:.2f} GB")
            
            # Show subdirectory breakdown
            if base_path.is_dir():
                for subdir in base_path.iterdir():
                    if subdir.is_dir():
                        subdir_size = self.get_directory_size(subdir)
                        if subdir_size > 0.1:  # Only show dirs > 100MB
                            print(f"   └── {subdir.name}: {subdir_size:.2f} GB")
        
        print(f"\n💾 Total storage: {total_size:.2f} GB")
        
        return dir_sizes
    
    def get_directory_size(self, path: Path) -> float:
        """Get directory size in GB"""
        total_size = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except PermissionError:
            pass
        return total_size / (1024**3)
    
    def clean_huggingface_cache(self, dry_run: bool = True):
        """Clean up HuggingFace cache directories"""
        cache_dirs = [
            Path.home() / ".cache" / "huggingface",
            Path("./models/cache"),
            Path("./.cache")
        ]
        
        total_cleaned = 0
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                cache_size = self.get_directory_size(cache_dir)
                total_cleaned += cache_size
                
                print(f"🧹 {'[DRY RUN] ' if dry_run else ''}Clean cache: {cache_dir} ({cache_size:.2f} GB)")
                
                if not dry_run and cache_size > 0:
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"   ✅ Removed {cache_dir}")
                    except Exception as e:
                        print(f"   ❌ Failed to remove: {e}")
        
        return total_cleaned
    
    def consolidate_model_files(self, model_dir: str, dry_run: bool = True):
        """Consolidate split model files where possible"""
        model_path = Path(model_dir)
        if not model_path.exists():
            return
        
        print(f"\n🔧 Consolidating {model_dir}...")
        
        # Look for safetensors index files
        for index_file in model_path.rglob("*.safetensors.index.json"):
            print(f"   Found index: {index_file}")
            
            try:
                with open(index_file) as f:
                    index_data = json.load(f)
                
                weight_map = index_data.get("weight_map", {})
                unique_files = set(weight_map.values())
                
                if len(unique_files) > 1:
                    print(f"   📦 Model split into {len(unique_files)} files")
                    
                    # Check if we have all split files
                    split_dir = index_file.parent
                    missing_files = []
                    total_size = 0
                    
                    for split_file in unique_files:
                        split_path = split_dir / split_file
                        if split_path.exists():
                            total_size += split_path.stat().st_size
                        else:
                            missing_files.append(split_file)
                    
                    if missing_files:
                        print(f"   ⚠️  Missing {len(missing_files)} split files")
                        for missing in missing_files[:3]:  # Show first 3
                            print(f"      - {missing}")
                        if len(missing_files) > 3:
                            print(f"      ... and {len(missing_files) - 3} more")
                    else:
                        print(f"   ✅ All split files present ({total_size / 1024**3:.2f} GB total)")
                        
                        # Note: Actual consolidation would require understanding the tensor structure
                        # This is a placeholder for more advanced consolidation logic
                        
            except Exception as e:
                print(f"   ❌ Error reading index: {e}")
    
    def optimize_all(self, dry_run: bool = True):
        """Run all optimization steps"""
        print("🚀 AutoStudio Model Optimizer")
        print("=" * 50)
        
        # Define search directories
        search_dirs = [
            "./models",
            "./models_efficient", 
            "./.cache",
            str(Path.home() / ".cache" / "huggingface")
        ]
        
        # Step 1: Analyze current storage
        print("\n1️⃣ Storage Analysis")
        self.analyze_model_storage(search_dirs)
        
        # Step 2: Find duplicates
        print("\n2️⃣ Duplicate Detection")
        duplicates = self.find_duplicate_files(search_dirs)
        
        if duplicates:
            saved_gb = self.remove_duplicates(duplicates, dry_run=dry_run)
            print(f"\n💾 Potential savings: {saved_gb:.2f} GB")
        else:
            print("✅ No duplicates found")
        
        # Step 3: Clean cache
        print("\n3️⃣ Cache Cleanup")
        cache_cleaned = self.clean_huggingface_cache(dry_run=dry_run)
        print(f"🧹 Cache to clean: {cache_cleaned:.2f} GB")
        
        # Step 4: Model consolidation analysis
        print("\n4️⃣ Model File Analysis")
        for search_dir in ["./models", "./models_efficient"]:
            if Path(search_dir).exists():
                self.consolidate_model_files(search_dir, dry_run=dry_run)
        
        total_savings = saved_gb + cache_cleaned
        print(f"\n🎉 Total potential savings: {total_savings:.2f} GB")
        
        if dry_run:
            print("\n⚠️  This was a dry run. Use --execute to apply changes.")

def main():
    """Main optimization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize AutoStudio model storage")
    parser.add_argument("--execute", action="store_true", help="Actually perform optimizations (default is dry run)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze storage, don't optimize")
    
    args = parser.parse_args()
    
    optimizer = ModelOptimizer()
    
    if args.analyze_only:
        # Just analyze storage
        search_dirs = ["./models", "./models_efficient", "./.cache"]
        optimizer.analyze_model_storage(search_dirs)
    else:
        # Run full optimization
        optimizer.optimize_all(dry_run=not args.execute)

if __name__ == "__main__":
    main()