import argparse
import sys
import os

"""
input file:
test/PUCPR-Cloudy-2012-10-05_08_37_50.jpg
test/PUCPR-Cloudy-2012-10-05_08_42_50.jpg
test/PUCPR-Cloudy-2012-10-05_08_47_50.jpg

move the files from current folder to test folder
"""

def mover(file_list):
    with open(file_list, 'r') as f:
        data = f.readlines()

    # remove the newline character
    data = [x.strip() for x in data]

    for image in data:
        parts = image.split('/')
        dir_name = parts[0]
        file_name = parts[1]

        # move the file
        print(f"Moving images/{file_name} to {dir_name}")
        os.rename(os.path.join("images", file_name), os.path.join(dir_name, file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move files to directory')

    parser.add_argument('text_file', type=str, help='The text file')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    mover(args.text_file)