import subprocess
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="fer2013", type=str, help="fer2013 or ferplus")
args = vars(parser.parse_args())

def install_package(package):
    subprocess.check_call(["pip", "install", package])

def download_file(file_id, output_name):
    install_package("gdown")
    import gdown
    gdown.download(id=file_id, output=output_name, quiet=False)

def unzip_file(zip_file):
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_file)

def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


if args['dataset'] == "fer2013":
    download_file('1YBuZaO7morIG43trYi0qtdelYGukBNCj', 'FER2013.zip')
    unzip_file('FER2013.zip')
elif args['dataset'] == "ferplus":
    download_file('1LShk6tZlsdBO-DciChK7y7nivUOvTAFk', 'FERPlus.zip')
    unzip_file('FERPlus.zip')
    remove_directory('fer_plus/train/contempt')
    remove_directory('fer_plus/val/contempt')
    remove_directory('fer_plus/test/contempt')