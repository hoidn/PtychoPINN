import os
from ptycho.xpp import get_data
from ptycho.loader import RawData

def main():
    # Load RawData instances using the 'xpp' method
    train_data, test_data = get_data()

    # Define file paths for output
    train_data_file_path = 'train_data.npz'
    test_data_file_path = 'test_data.npz'

    # Use RawData.to_file() to write them to file
    train_data.to_file(train_data_file_path)
    test_data.to_file(test_data_file_path)

    print(f"Train data written to {train_data_file_path}")
    print(f"Test data written to {test_data_file_path}")

if __name__ == '__main__':
    main()
