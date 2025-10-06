import os
import gdown
import zipfile
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# CORRECTED URLs - Updated for proper Google Drive access
# DistilBERT and RoBERTa now use pickle files instead of folder downloads
MODEL_URLS = {
    "xgboost": "https://drive.google.com/uc?id=1q9LIMFBtJ-VxNaXQ9ajoQ9af-rvFZyak&export=download",
    "svm": "https://drive.google.com/uc?id=1cObIWea7gket7gpd76jEkpsZYZ_rqRDe&export=download", 
    "distilbert_v2": "https://drive.google.com/uc?id=1UvQbDXenO2uLDlxQVJvEAMhmMZUC3OHP&export=download",
    "roberta": "https://drive.google.com/uc?id=1h6hpo9MsOUsA8Cmp-wHh0YRSPAmLOB2m&export=download"
}

class ModelDownloader:
    def __init__(self):
        self.base_path = Path("/app/models")
        self.setup_directories()
        
    def setup_directories(self):
        """Create directories with proper permissions"""
        self.base_path.mkdir(exist_ok=True)
        
        # Fix gdown cache directory issue
        cache_dir = Path("/tmp/gdown_cache")
        cache_dir.mkdir(exist_ok=True)
        os.environ['GDOWN_CACHE_DIR'] = str(cache_dir)
        
    def download_model(self, model_name):
        """Download model with multiple fallback methods"""
        try:
            model_paths = {
                "xgboost": self.base_path / "xgboost_model.pkl",
                "svm": self.base_path / "svm_model.pkl",
                "distilbert_v2": self.base_path / "distilbert_v2_model.pkl",  # Now a pickle file
                "roberta": self.base_path / "roberta_model.pkl"  # Now a pickle file
            }
            
            model_path = model_paths[model_name]
            
            # Skip if already exists
            if self._check_model_exists(model_path, model_name):
                logger.info(f"✅ {model_name} already exists")
                return True
                
            logger.info(f"⬇️ Downloading {model_name}...")
            
            url = MODEL_URLS[model_name]
            temp_dir = Path("/tmp/model_downloads")
            temp_dir.mkdir(exist_ok=True, mode=0o777)
            temp_file = temp_dir / f"{model_name}_download.tmp"
            
            # All models are now single files (pickle format)
            logger.info(f"📄 {model_name} is a single file model")
            
            # Method 1: Try gdown with custom cache
            if self._download_with_gdown_fixed(url, temp_file):
                self._move_file_safely(temp_file, model_path)
                logger.info(f"✅ Downloaded {model_name}")
                return True
            else:
                logger.warning(f"⚠️ GDown failed for {model_name}, trying requests...")
                # Method 2: Try direct download with requests
                if self._download_with_requests(url, temp_file):
                    self._move_file_safely(temp_file, model_path)
                    logger.info(f"✅ Downloaded {model_name} via requests")
                    return True
            
            logger.error(f"❌ All download methods failed for {model_name}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def _move_file_safely(self, source, destination):
        """Safely move file with proper permissions"""
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            # Copy file instead of move to avoid permission issues
            import shutil
            shutil.copy2(str(source), str(destination))
            
            # Remove source file
            source.unlink()
            
            # Set proper permissions
            destination.chmod(0o666)
            
            return True
        except Exception as e:
            logger.error(f"Failed to move file safely: {e}")
            return False
    
    def _download_with_gdown_fixed(self, url, output_path):
        """Download using gdown with fixed cache directory"""
        try:
            # Set custom cache directory
            cache_dir = Path("/tmp/gdown_cache")
            cache_dir.mkdir(exist_ok=True, mode=0o777)
            
            # Download without relying on gdown's cache - remove fuzzy for better compatibility
            gdown.download(url, str(output_path), quiet=False, use_cookies=False)
            
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            return False
        except Exception as e:
            logger.warning(f"GDown fixed method failed: {e}")
            return False
    
    def _download_with_requests(self, url, output_path):
        """Download using requests as fallback"""
        try:
            session = requests.Session()
            
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return output_path.exists() and output_path.stat().st_size > 0
            
        except Exception as e:
            logger.warning(f"Requests download failed: {e}")
            return False
    
    def _download_folder_from_drive(self, url, temp_file, model_path, model_name):
        """Download folder from Google Drive as zip and extract"""
        try:
            logger.info(f"📁 Attempting to download {model_name} folder from Google Drive")
            
            # Method 1: Try gdown with folder download
            if self._download_with_gdown_fixed(url, temp_file):
                return self._extract_folder_zip(temp_file, model_path, model_name)
            
            # Method 2: Try requests
            if self._download_with_requests(url, temp_file):
                return self._extract_folder_zip(temp_file, model_path, model_name)
            
            # Method 3: Try alternative download methods for Google Drive folders
            return self._download_drive_folder_alternative(url, temp_file, model_path, model_name)
            
        except Exception as e:
            logger.error(f"Failed to download folder {model_name}: {e}")
            return False
    
    def _download_drive_folder_alternative(self, url, temp_file, model_path, model_name):
        """Alternative method to download Google Drive folders"""
        try:
            # Extract file ID from URL
            import re
            id_match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
            if not id_match:
                return False
            
            file_id = id_match.group(1)
            
            # Try downloading with gdown using folder ID
            import gdown
            
            # Create temporary directory for folder download
            temp_folder = Path(f"/tmp/{model_name}_folder")
            temp_folder.mkdir(exist_ok=True, mode=0o777)
            
            try:
                # Download folder directly
                gdown.download_folder(f"https://drive.google.com/drive/folders/{file_id}", 
                                    str(temp_folder), quiet=False, use_cookies=False)
                
                # Move contents to model path
                if temp_folder.exists() and any(temp_folder.iterdir()):
                    model_path.mkdir(parents=True, exist_ok=True, mode=0o777)
                    
                    # Copy all files from temp folder to model path
                    import shutil
                    for item in temp_folder.iterdir():
                        if item.is_file():
                            shutil.copy2(item, model_path)
                        elif item.is_dir():
                            shutil.copytree(item, model_path / item.name, dirs_exist_ok=True)
                    
                    # Clean up temp folder
                    shutil.rmtree(temp_folder)
                    
                    logger.info(f"✅ Successfully downloaded {model_name} folder")
                    return True
            except Exception as e:
                logger.warning(f"Folder download attempt failed: {e}")
                
                # Clean up on failure
                if temp_folder.exists():
                    import shutil
                    shutil.rmtree(temp_folder, ignore_errors=True)
                    
            return False
            
        except Exception as e:
            logger.error(f"Alternative folder download failed: {e}")
            return False
    
    def _extract_folder_zip(self, zip_path, extract_to, model_name):
        """Extract zip file for folder models"""
        try:
            extract_to.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            # Check if it's actually a zip file
            if not zipfile.is_zipfile(zip_path):
                logger.warning(f"Downloaded file for {model_name} is not a zip file")
                return False
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to.parent)
            
            # Set permissions for extracted files
            if extract_to.exists():
                for root, dirs, files in os.walk(extract_to):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
            
            zip_path.unlink()
            logger.info(f"✅ Successfully extracted {model_name} folder")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract folder zip for {model_name}: {e}")
            return False
    
    def _extract_zip(self, zip_path, extract_to):
        """Extract zip file with proper permissions"""
        try:
            extract_to.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to.parent)
            
            # Set permissions for extracted files
            if extract_to.exists():
                for root, dirs, files in os.walk(extract_to):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
            
            zip_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to extract zip: {e}")
            return False
    
    def _check_model_exists(self, model_path, model_name):
        """Check if model exists"""
        # All models are now single pickle files
        return model_path.exists() and model_path.stat().st_size > 0
    
    def download_all_models(self):
        """Download all models"""
        models = ['xgboost', 'svm', 'distilbert_v2', 'roberta']
        success_count = 0
        
        for model in models:
            if self.download_model(model):
                success_count += 1
        
        logger.info(f"📊 Download complete: {success_count}/{len(models)} models")
        return success_count > 0

downloader = ModelDownloader()