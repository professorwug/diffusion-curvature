#!/usr/bin/env python3
"""
Cloud data synchronization script for aiconfi project.

This script provides functionality to:
1. Sync zip files from local data directory to cloud storage with timestamps
2. List and selectively sync zip files from cloud storage to local data directory

Usage:
    python sync_cloud_data.py sync-to-cloud
    python sync_cloud_data.py sync-from-cloud
"""

import argparse
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv


class CloudDataSync:
    """Handles synchronization of zip files between local and cloud storage."""
    
    def __init__(self):
        load_dotenv()
        self.cloud_storage_dir = os.getenv('CLOUD_STORAGE_DIR')
        if not self.cloud_storage_dir:
            raise ValueError("CLOUD_STORAGE_DIR environment variable not set")
        
        self.local_data_dir = Path(__file__).parent.parent / "data"
        self.cloud_aiconfi_dir = Path(self.cloud_storage_dir) / "aiconfi"
        
        # Ensure directories exist
        self.local_data_dir.mkdir(exist_ok=True)
        self.cloud_aiconfi_dir.mkdir(parents=True, exist_ok=True)
    
    def get_timestamp(self) -> str:
        """Generate a timestamp string for versioning."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def find_local_zip_files(self) -> List[Path]:
        """Find all zip files in the local data directory."""
        return list(self.local_data_dir.glob("**/*.zip"))
    
    def find_cloud_zip_files(self) -> List[Path]:
        """Find all zip files in the cloud storage directory."""
        return list(self.cloud_aiconfi_dir.glob("**/*.zip"))
    
    def extract_base_filename(self, zip_path: Path) -> str:
        """Extract the base filename without timestamp and extension."""
        filename = zip_path.stem
        # Remove timestamp pattern if present (YYYYMMDD_HHMMSS)
        if len(filename) >= 15 and filename[-15:-8] == "_" and filename[-8:].isdigit():
            if filename[-15:-8].replace("_", "").isdigit():
                return filename[:-16]  # Remove _YYYYMMDD_HHMMSS
        return filename
    
    def get_file_modification_time(self, file_path: Path) -> datetime:
        """Get the modification time of a file."""
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def sync_to_cloud(self) -> None:
        """Sync local zip files to cloud storage with timestamp versioning."""
        local_zips = self.find_local_zip_files()
        
        if not local_zips:
            print("No zip files found in local data directory.")
            return
        
        print(f"Found {len(local_zips)} zip file(s) in local data directory:")
        
        for local_zip in local_zips:
            timestamp = self.get_timestamp()
            base_name = local_zip.stem
            new_name = f"{base_name}_{timestamp}.zip"
            cloud_path = self.cloud_aiconfi_dir / new_name
            
            print(f"  Copying {local_zip.name} -> {cloud_path.name}")
            shutil.copy2(local_zip, cloud_path)
        
        print(f"Successfully synced {len(local_zips)} file(s) to cloud storage.")
    
    def get_newer_cloud_files(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        Get cloud files that are newer than local versions or don't exist locally.
        
        Returns:
            List of tuples: (cloud_file_path, local_file_path or None)
        """
        cloud_zips = self.find_cloud_zip_files()
        local_zips = self.find_local_zip_files()
        
        # Create mapping of base names to local files
        local_base_map = {}
        for local_zip in local_zips:
            base_name = self.extract_base_filename(local_zip)
            if base_name not in local_base_map:
                local_base_map[base_name] = []
            local_base_map[base_name].append(local_zip)
        
        newer_files = []
        
        for cloud_zip in cloud_zips:
            base_name = self.extract_base_filename(cloud_zip)
            cloud_mtime = self.get_file_modification_time(cloud_zip)
            
            if base_name not in local_base_map:
                # File doesn't exist locally
                newer_files.append((cloud_zip, None))
            else:
                # Check if cloud file is newer than any local version
                local_files = local_base_map[base_name]
                newest_local_mtime = max(self.get_file_modification_time(f) for f in local_files)
                
                if cloud_mtime > newest_local_mtime:
                    # Find the newest local file to replace
                    newest_local = max(local_files, key=self.get_file_modification_time)
                    newer_files.append((cloud_zip, newest_local))
        
        return newer_files
    
    def sync_from_cloud(self) -> None:
        """Interactive sync of newer cloud files to local data directory."""
        newer_files = self.get_newer_cloud_files()
        
        if not newer_files:
            print("No newer files found in cloud storage.")
            return
        
        print(f"Found {len(newer_files)} newer file(s) in cloud storage:")
        print()
        
        # Display options
        for i, (cloud_file, local_file) in enumerate(newer_files, 1):
            cloud_mtime = self.get_file_modification_time(cloud_file)
            if local_file:
                local_mtime = self.get_file_modification_time(local_file)
                print(f"{i}. {cloud_file.name}")
                print(f"   Cloud: {cloud_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Local: {local_mtime.strftime('%Y-%m-%d %H:%M:%S')} ({local_file.name})")
            else:
                print(f"{i}. {cloud_file.name} (NEW)")
                print(f"   Cloud: {cloud_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # Interactive selection
        while True:
            try:
                selection = input(f"Select files to sync (1-{len(newer_files)}, 'a' for all, 'q' to quit): ").strip()
                
                if selection.lower() == 'q':
                    print("Sync cancelled.")
                    return
                
                if selection.lower() == 'a':
                    selected_indices = list(range(len(newer_files)))
                    break
                
                # Parse comma-separated indices
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                if all(0 <= i < len(newer_files) for i in indices):
                    selected_indices = indices
                    break
                else:
                    print(f"Invalid selection. Please enter numbers between 1 and {len(newer_files)}.")
            
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please try again.")
        
        # Sync selected files
        synced_count = 0
        for i in selected_indices:
            cloud_file, local_file = newer_files[i]
            
            # Determine local destination
            base_name = self.extract_base_filename(cloud_file)
            local_dest = self.local_data_dir / f"{base_name}.zip"
            
            # Create subdirectory if needed (preserve relative structure)
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Copying {cloud_file.name} -> {local_dest.name}")
            shutil.copy2(cloud_file, local_dest)
            
            # If it's a zip file, offer to extract
            if self.should_extract_zip(local_dest):
                self.extract_zip_file(local_dest)
            
            synced_count += 1
        
        print(f"Successfully synced {synced_count} file(s) from cloud storage.")
    
    def should_extract_zip(self, zip_path: Path) -> bool:
        """Ask user if they want to extract the zip file."""
        while True:
            response = input(f"Extract {zip_path.name}? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def extract_zip_file(self, zip_path: Path) -> None:
        """Extract a zip file to the same directory."""
        extract_dir = zip_path.parent / zip_path.stem
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file.")
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Sync zip files between local and cloud storage")
    parser.add_argument(
        'action',
        choices=['sync-to-cloud', 'sync-from-cloud'],
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    try:
        sync = CloudDataSync()
        
        if args.action == 'sync-to-cloud':
            sync.sync_to_cloud()
        elif args.action == 'sync-from-cloud':
            sync.sync_from_cloud()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())