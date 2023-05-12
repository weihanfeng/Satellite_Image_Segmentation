import splitfolders
import os
import shutil

class train_test_split:
    def __init__(self, seed, train_ratio, val_ratio, test_ratio) -> None:
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_folder(self, input_dir, output_dir):
        # create output folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # print number of subfolders and respective number of files in input dir
        print("Number of subfolders and respective number of files in input dir:")
        for root, sub_dirs, files in os.walk(input_dir):
            print(root, ": ", len(files))

        splitfolders.ratio(
            input_dir,
            output_dir,
            seed=self.seed,
            ratio=(self.train_ratio, self.val_ratio, self.test_ratio),
            move=True,
        )
        # print number of subfolders and respective number of files in output dir
        print("Number of subfolders and respective number of files in output dir:")
        for root, sub_dirs, files in os.walk(output_dir):
            print(root, ": ", len(files))
        # Print folder structure of output dir
        print("Folder structure of output dir:")
        for root, sub_dirs, files in os.walk(output_dir):
            print(root, ": ", sub_dirs)
        
        # remove input dir
        shutil.rmtree(input_dir)

if __name__ == "__main__":
    input_dir = "data/selected_data"
    output_dir = "data/selected_data_split"
    seed = 42
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    train_test_split = train_test_split(seed, train_ratio, val_ratio, test_ratio)
    train_test_split.split_folder(input_dir, output_dir)
