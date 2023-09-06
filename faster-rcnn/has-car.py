import json


if __name__ == '__main__':
    # set the dataset root
    root = 'datasets/pklot/images/PUCPR/test'

    # load json
    with open(root + '/cars.json', 'r') as f:
        data = json.load(f)

    # all image ids
    image_ids = [a["id"] for a in data["images"]]

    # image ids with annotions
    ann_ids = set([ a["image_id"] for a in data["annotations"] ])

    # image ids without annotions
    image_ids_without_anns = set(image_ids) - ann_ids

    # filter images without annotations
    data["images"] = [ a for a in data["images"] if not a["id"] in image_ids_without_anns ]

    # write to file
    with open(root + '/has-cars.json', 'w') as f:
        json.dump(data, f, indent=4)