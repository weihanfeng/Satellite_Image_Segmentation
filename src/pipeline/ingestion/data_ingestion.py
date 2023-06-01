"""
Module to download dataset as zip file and extract it to folder
"""
import os
import zipfile
import requests
from tqdm import tqdm
import gdown
import shutil

class DataIngestion:
    def __init__(self, url, file_path, extract_path):
        self.url = url
        self.file_path = file_path
        self.extract_path = extract_path


    def download_data(self, download_from_gdrive):
        """
        Download data from url
        """
        if download_from_gdrive:
                gdown.download(output=self.file_path, quiet=False, id = self.url)
        else: 
            response = requests.get(self.url, allow_redirects=True, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading data")
            with open(self.file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")

    def extract_data(self):
        """
        Extract data from zip file
        """
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)

    def move_files(self, src_paths: list, dest: str):
        """
        Move files from src dir to destination dir. 
        Create destination dir if it doesn't exist. 
        If file already exists in destination dir, it will be overwritten.
        """
        if not os.path.exists(dest):
            os.makedirs(dest)
        for src_path in src_paths:
            for file in os.listdir(src_path):
                # if file exists in destination dir, overwrite it
                os.replace(os.path.join(src_path, file), os.path.join(dest, file))


    def get_size(self):
        """Get size of downloaded zip file in gb"""
        return os.path.getsize(self.file_path) / (1024 * 1024 * 1024)

if __name__ == "__main__":
    URL = "1xbnKVN5aRMlpxISXgutzQO0hPT_b4lMi"
    FILE_PATH = os.path.join(os.getcwd(), "data", "data.zip")
    EXTRACT_PATH = os.path.join(os.getcwd(), "data")
    DATA_INGESTION = DataIngestion(URL, FILE_PATH, EXTRACT_PATH)
    # DATA_INGESTION.download_data(download_from_gdrive=True)
    # DATA_INGESTION.extract_data()
    src_paths = ["data/Train/Rural/images_png", "data/Train/Urban/images_png"]
    dest = "data/images"
    DATA_INGESTION.move_files(src_paths, dest)
    
