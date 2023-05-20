from pipeline.ingestion.data_ingestion import DataIngestion
from pipeline.processing.data_split import DataSplit
from pipeline.processing.train_test_split import TRAIN_TEST_SPLIT
import hydra
from omegaconf import DictConfig
import os
import shutil


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def ingest_and_process_data(cfg: DictConfig):
    """
    Download data, split image and perform train-test-val split
    """
    # Ingest data if FLAG is True
    if cfg["data_ingestion"]["FLAG"]:
        URL = cfg["data_ingestion"]["URL"]
        FILE_PATH = os.path.join(
            os.getcwd(), cfg["files"]["DATA_DIR"], cfg["files"]["FILE_NAME"]
        )
        EXTRACT_PATH = os.path.join(os.getcwd(), cfg["files"]["DATA_DIR"])
        DATA_INGESTION = DataIngestion(URL, FILE_PATH, EXTRACT_PATH)
        DATA_INGESTION.download_data()
        DATA_INGESTION.extract_data()

    # Split image
    data_split = DataSplit(
        patch_size=cfg["data_split"]["split_image"]["PATCH_SIZE"],
        image_dir=cfg["files"]["DL_IMG_DIR"],
        mask_dir=cfg["files"]["DL_MASK_DIR"],
        output_dir=cfg["files"]["INTERIM_DIR"],
        selection_threshold=0.95,
    )
    data_split._split_and_select_patches()

    # Train-test-val split
    train_test_split = TRAIN_TEST_SPLIT(
        seed=cfg["data_split"]["train_val_split"]["RANDOM_STATE"],
        train_ratio=cfg["data_split"]["train_val_split"]["TRAIN_SIZE"],
        val_ratio=cfg["data_split"]["train_val_split"]["VAL_SIZE"],
    )
    train_test_split.split_folder(
        input_dir=cfg["files"]["INTERIM_DIR"], output_dir=cfg["files"]["SPLIT_DIR"]
    )

    # Remove DL dir
    shutil.rmtree(cfg["files"]["DL_IMG_DIR"])
    shutil.rmtree(cfg["files"]["DL_MASK_DIR"])
    os.remove(os.path.join(cfg["files"]["DATA_DIR"], cfg["files"]["FILE_NAME"]))


if __name__ == "__main__":
    ingest_and_process_data()
