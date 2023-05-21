import splitfolders
import os
import shutil

class TRAIN_TEST_SPLIT:
    def __init__(self, seed, train_ratio, val_ratio) -> None:
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1-train_ratio-val_ratio

    def split_folder(self, input_dir, output_dir):
        # create output folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        splitfolders.ratio(
            input_dir,
            output_dir,
            seed=self.seed,
            ratio=(self.train_ratio, self.val_ratio, self.test_ratio),
            move=True,
        )
        # remove input dir
        shutil.rmtree(input_dir)

if __name__ == "__main__":
    input_dir = "data/selected_data"
    output_dir = "data/selected_data_split"
    seed = 42
    train_ratio = 0.7
    val_ratio = 0.15
    train_test_split = TRAIN_TEST_SPLIT(seed, train_ratio, val_ratio)
    train_test_split.split_folder(input_dir, output_dir)
