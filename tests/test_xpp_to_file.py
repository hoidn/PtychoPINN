import os
from ptycho.xpp import ptycho_data
from ptycho import loader

def main():
    # Load RawData instances using the 'xpp' method
    train_data = ptycho_data

    # Define file paths for output
    train_data_file_path = 'train_data.npz'

    # Use RawData.to_file() to write them to file
    train_data.to_file(train_data_file_path)

    print(f"Train data written to {train_data_file_path}")

if __name__ == '__main__':
    main()
