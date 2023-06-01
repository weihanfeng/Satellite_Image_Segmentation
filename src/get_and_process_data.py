from pipeline.ingestion.data_ingestion import DataIngestion
from pipeline.processing.data_split import DataSplit
from pipeline.processing.train_test_split import TRAIN_TEST_SPLIT
import hydra
from omegaconf import DictConfig
import os
import shutil
import logging
from utils.general_utils import get_num_files, setup_logging

setup_logging(logging_config_path="./conf/base/logging.yaml")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def ingest_and_process_data(cfg: DictConfig):
    """
    Download data, split image and perform train-test-val split
    """
    # Ingest data if FLAG is True
    logging.info("Ingesting data...")
    if cfg["data_ingestion"]["FLAG"]:
        URL = cfg["data_ingestion"]["URL"]
        FILE_PATH = os.path.join(
            os.getcwd(), cfg["files"]["DATA_DIR"], cfg["files"]["FILE_NAME"]
        )
        EXTRACT_PATH = os.path.join(os.getcwd(), cfg["files"]["DATA_DIR"])
        DATA_INGESTION = DataIngestion(URL, FILE_PATH, EXTRACT_PATH)
        if cfg["data_ingestion"]["DOWNLOAD_FLAG"]:
            logging.info("Start downloading data...")
            DATA_INGESTION.download_data(cfg["data_ingestion"]["DOWNLOAD_FROM_GDRIVE"])
        logging.info("Start extracting data...")
        DATA_INGESTION.extract_data()
        logging.info("Moving files from DL dirs to DEST dirs...")
        DATA_INGESTION.move_files(
            src_paths=cfg["files"]["DL_IMG_DIRS"],
            dest=cfg["files"]["DEST_IMG_DIR"],
        )
        DATA_INGESTION.move_files(
            src_paths=cfg["files"]["DL_MASK_DIRS"],
            dest=cfg["files"]["DEST_MASK_DIR"],
        )
    logging.info(
            "Number of extracted images: "
            + str(get_num_files(cfg["files"]["DEST_IMG_DIR"]))
        )

    # Split image
    logging.info(
        f"Splitting images into image patches of size {cfg['data_split']['split_image']['PATCH_SIZE']}..."
    )
    data_split = DataSplit(
        patch_size=cfg["data_split"]["split_image"]["PATCH_SIZE"],
        image_dir=cfg["files"]["DEST_IMG_DIR"],
        mask_dir=cfg["files"]["DEST_MASK_DIR"],
        output_dir=cfg["files"]["INTERIM_DIR"],
        labels_to_remove=cfg["data_split"]["split_image"]["LABELS_TO_REMOVE"],
        selection_threshold=cfg["data_split"]["split_image"]["SELECTION_THRESHOLD"],
        new_label_map=cfg["data_split"]["split_image"]["NEW_LABEL_MAP"],
    )
    data_split.split_and_select_patches()

    logging.info("Splitting into train-test-val dir...")
    # Train-test-val split
    train_test_split = TRAIN_TEST_SPLIT(
        seed=cfg["data_split"]["train_val_split"]["RANDOM_STATE"],
        train_ratio=cfg["data_split"]["train_val_split"]["TRAIN_SIZE"],
        val_ratio=cfg["data_split"]["train_val_split"]["VAL_SIZE"],
    )
    train_test_split.split_folder(
        input_dir=cfg["files"]["INTERIM_DIR"], output_dir=cfg["files"]["SPLIT_DIR"]
    )

    # Remove DL files
    logging.info("Removing downloaded and interim data...")
    # If file or dir is not SPLIT_DIR, remove it
    for file in os.listdir(cfg["files"]["DATA_DIR"]):
        file_txt = os.path.join(cfg["files"]["DATA_DIR"], file).replace("/", "")
        # make SPLIT_DIR into a path
        split_dir_txt = os.path.join(cfg["files"]["SPLIT_DIR"]).replace("/", "")
        if (file_txt != split_dir_txt) and (file != ".gitignore"):
            # join file path to DATA_DIR
            if os.path.isfile(os.path.join(cfg["files"]["DATA_DIR"], file)):
                os.remove(os.path.join(cfg["files"]["DATA_DIR"], file))
            else:
                shutil.rmtree(os.path.join(cfg["files"]["DATA_DIR"], file))
    
    logging.info("Data ingestion and processing completed.")
    
if __name__ == "__main__":
    ingest_and_process_data()
