import gdown
import os


def download_pyg_dataset(output_dir=None):
    """Download Stanford preprocessed PyG dataset from Google Drive.

    Args:
        output_dir: Directory to save the dataset. Defaults to script directory.
    """
    google_dir = "https://drive.google.com/drive/folders/1QANENxeWRVBs2TZ8SQ5CGuHo27i95WtO?usp=sharing"
    if output_dir is None:
        output_dir = os.path.dirname(os.path.realpath(__file__))
    print("Attempting download...")
    gdown.download_folder(
        url=google_dir, quiet=True, use_cookies=False, output=output_dir
    )
    print("Finished!")
    os.rename(str(output_dir) + "datasets", str(output_dir) + "cylinder_flow_stanford")


if __name__ == "__main__":
    download_pyg_dataset()
