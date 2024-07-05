""" rename the to_be_removed.txt """
import sys
import argparse
from tqdm import tqdm
from renamer import extract_date_from_target_name, create_target_name_dictionary, extract_date_from_src_name

def do_rename(text_file, mapper, output):
    # read the file list
    with open(text_file, 'r') as f:
        data = f.readlines()

    with open(output, 'w') as f:
        for image in tqdm(data):
            date = extract_date_from_src_name(image.strip())
            target_name = mapper[date]
            f.write(f'{target_name}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rename files in coco annotation')
    parser.add_argument('text_file', type=str, help='The text file')
    parser.add_argument('target_name_list_file', type=str, help='The target name list file')
    parser.add_argument('output', type=str, help='The output file')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    mapper = create_target_name_dictionary(args.target_name_list_file)

    do_rename(args.text_file, mapper, args.output)