"""
Module to download dataset as zip file and extract it to folder
"""
import os
import zipfile
import requests
from tqdm import tqdm

class DataIngestion:
    def __init__(self, url, file_path, extract_path):
        self.url = url
        self.file_path = file_path
        self.extract_path = extract_path


    def download_data(self):
        """
        Download data from url
        """
        response = requests.get(self.url, stream=True)
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

if __name__ == "__main__":
    URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
    FILE_PATH = os.path.join(os.getcwd(), "data", "data.zip")
    EXTRACT_PATH = os.path.join(os.getcwd(), "data")
    DATA_INGESTION = DataIngestion(URL, FILE_PATH, EXTRACT_PATH)
    DATA_INGESTION.download_data()
    DATA_INGESTION.extract_data()
