""" Convert CNR csv format to COCO json """

import csv
import os
import datetime
import time
import json
import glob
import PIL
import PIL.Image
from PIL.ExifTags import TAGS

ROOT = './datasets/cnr/'
PARENT = 'FULL_IMAGE_1000x750/'
WEATHER = {
    'S': 'SUNNY',
    'O': 'OVERCAST',
    'R': 'RAINY'
}

class CocoAnnotaionGenerator:
    """ Generate coco annotation file for a list of images and bounding boxes """

    def __init__(self, annotation_path):
        self.annotation_path = annotation_path
        self.images = {} # image path to id
        self.annotations = []
        self.root = {}
        self.root['info'] = {
            "description": "PKLot dataset - full annotation",
            "url": "https://app.roboflow.com/personal-ysmui/pklot-full-annotation",
            "version": "1.0",
            "year": 2023,
            "contributor": "University of Buckingham",
            "date_created": "2023-07-04T09:00:00+00:00"
        }
        self.root['licenses'] = [
            {
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "id": 1,
                "name": "Attribution License"
            }
        ]
        self.root['categories'] = [
            {
                "id": 0,
                "name": "spaces",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "space-empty",
                "supercategory": "spaces"
            },
            {
                "id": 2,
                "name": "space-occupied",
                "supercategory": "spaces"
            }
        ]
        self.root['images'] = []
        self.root['annotations'] = []

    def add_annotation(self, image_path: str, bbox: list, category_id: int):
        """ Add an annotation to the annotations """
        if image_path not in self.images.keys():
            return

        image_id = self.images[image_path]

        self.root['annotations'].append({
            "id": len(self.root['annotations']),
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "iscrowd": 0
        })

    def add_image(self, image_path):
        """ Add an image to the annotations """

        if image_path in self.images.keys():
            return

        if not os.path.isfile(image_path):
            return

        print(image_path)

        image = PIL.Image.open(image_path)

        id = len(self.root['images'])
        # add the image
        self.root['images'].append({
            "id": id,
            "width": image.width,
            "height": image.height,
            "file_name": image_path,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": self.get_timestamp(image, image_path)
        })
        self.images[image_path] = id
        return id

    def get_timestamp(self, image, image_path):
        # Get the EXIF metadata
        exif_data = image._getexif()

        # Check if EXIF data is available
        if exif_data is not None:
            # Iterate over the EXIF tags
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    # The timestamp is stored in the "DateTimeOriginal" tag
                    return value

        # If no EXIF data is available, use the file creation time
        creation_time = os.path.getctime(image.filename)
        time_struct = time.localtime(creation_time)
        # Format the struct_time into a string
        formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", time_struct)
        return formatted_time

    def save(self):
        with open(self.annotation_path, 'w') as f:
            json.dump(self.root, f, indent=4)

    def generate(self):
        """ Generate the annotations """
        # get the list of images
        images = glob.glob(os.path.join(self.image_path, "*.jpg"))
        print("found %d images under: %s" % (len(images), self.image_path))

        tt = 0

        # for each image
        for image_path in images:
            print("%d: %s" % (tt, image_path))
            # load the image
            image = PIL.Image.open(image_path)

            # add the image to the annotations
            image_id = self.add_image(image, image_path)

            # convert the image to a tensor
            image = transforms.ToTensor()(image)

            # get the patches
            patches = get_patches(self.bboxes, image)

            # create a dataset for the patches
            dataset = PatchesDataset(patches)

            # create a dataloader for the dataset
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

            bbox_index = 0

            for imgs in dataloader:

                # disable gradient calculation
                with torch.no_grad():
                    output = self.model(imgs)
                    #_, predicted_labels = torch.max(output, dim=1)
                    predicted_labels = torch.argmax(output, dim=1)

                # add the annotations
                for i in range(len(predicted_labels)):
                    # get the label
                    label = predicted_labels[i].item()

                    # get the bbox
                    bbox = self.bboxes[bbox_index]

                    # add the annotation
                    self.add_annotation(image_id, bbox, label)

                    bbox_index += 1

            tt += 1
#            if tt == 2:
#                break

        # save the annotations
        with open(self.annotation_path, 'w') as f:
            json.dump(self.root, f, indent=4)

def load_csv():
    csv_file = ROOT + '/CNRPark+EXT.csv'
    with open(csv_file, newline='') as csvfile:
        datareader = csv.DictReader(csvfile)
        i=0
        for row in datareader:
            if row['camera'] not in ['A','B']:
                yield row


def add_all_annotations(generator: CocoAnnotaionGenerator, slots:object):
    for row in load_csv():
        camera = int(row['camera'])
        weather = WEATHER[row['weather']]
        year = int(row['year'])
        month = int(row['month'])
        day = int(row['day'])
        hour = int(row['hour'])
        minute = int(row['minute'])
        slot = int(row['slot_id'])
        occupancy = int(row['occupancy'])
        file_name = "%i-%02i-%02i_%02i%02i.jpg" % (year, month, day, hour, minute)
        date = "%i-%02i-%02i" % (year, month, day)
        path = "%s/%s/camera%i/%s" % (weather, date, camera, file_name)
        full_path = ROOT + PARENT + path
        bbox = slots[slot]
        category = 2 if occupancy == 1 else 1
        generator.add_annotation(full_path, bbox, category)

def add_all_images(generator: CocoAnnotaionGenerator):
    for row in load_csv():
        camera = int(row['camera'])
        weather = WEATHER[row['weather']]
        year = int(row['year'])
        month = int(row['month'])
        day = int(row['day'])
        hour = int(row['hour'])
        minute = int(row['minute'])
        slot = int(row['slot_id'])
        occupancy = int(row['occupancy'])
        file_name = "%i-%02i-%02i_%02i%02i.jpg" % (year, month, day, hour, minute)
        date = "%i-%02i-%02i" % (year, month, day)
        path = "%s/%s/camera%i/%s" % (weather, date, camera, file_name)

        full_path = ROOT + PARENT + path
        generator.add_image(full_path)

def load_slots():
    slots = {}
    for i in range(1, 10):
        csv_file = ROOT + "camera%i" % (i) + ".csv"
        print(csv_file)
        with open(csv_file, newline='') as csvfile:
            datareader = csv.DictReader(csvfile)
            for row in datareader:
                slot = int(row['SlotId'])
                x = int(row['X'])
                y = int(row['Y'])
                w = int(row['W'])
                h = int(row['H'])
                slots[slot] = [x, y, w, h]
    return slots

if __name__ == '__main__':
    generator = CocoAnnotaionGenerator('cnr.json')
    add_all_images(generator)
    slots = load_slots()
    add_all_annotations(generator, slots)
    generator.save()
    #print(slots)
