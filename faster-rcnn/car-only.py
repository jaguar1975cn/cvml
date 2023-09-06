import json


if __name__ == '__main__':
    # set the dataset root
    root = 'datasets/pklot/images/PUCPR/test'

    with open(root + '/annotations.json', 'r') as f:
        data = json.load(f)

    annotations = data['annotations']

    data['annotations'] = list(
        filter(
            lambda x: x['category_id'] == 1,
            annotations
        )
    )

    with open(root + '/cars.json', 'w') as f:
        json.dump(data, f, indent=4)