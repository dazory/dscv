import os
import copy
import json
import argparse

from dscv.utils.utils import get_dir_and_file_name


class MiniCOCODataset:
    def __init__(self, data_root, ann_file, save_root, img_prefix):
        self.data_root = data_root
        self.save_root = save_root
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        with open(f"{data_root}/{ann_file}", 'r') as file:
            self.data = json.load(file)

    def get_mini_dataset(self, num_images=10):
        '''
        self.data
            > info = {dict: 6}
            > license = {list: 8}
            > images = {list: 10}
                > image = {dict: 8}
                    > license, file_name, coco_url, height, width, date_captured, flickr_url, id
            > annotations = {list: 130}
                > annotation = {dict: 7}
                    > segmentation, area, iscrowd, image_id, bbox, category_id, id
            > categories = {list: 26}
                > category = {dict: 3}
                    > supercategory, id, name
        '''
        mini_data = dict()

        # copy info and license
        mini_data['info'] = copy.deepcopy(self.data['info'])
        mini_data['licenses'] = copy.deepcopy(self.data['licenses'])

        # Slice images
        mini_data["images"] = copy.deepcopy(self.data['images'][:num_images])
        image_ids = [image['id'] for image in mini_data['images']]

        # Parse annotations
        mini_data['annotations'] = []
        for anno in self.data['annotations']:
            if anno['image_id'] in image_ids:
                mini_data['annotations'].append(anno)

        category_ids = list(set([anno['category_id'] for anno in mini_data['annotations']])) # remove duplicates

        # Parse categories
        mini_data['categories'] = []
        for cate in self.data['categories']:
            if cate['id'] in category_ids:
                mini_data['categories'].append(cate)

        return mini_data

    def save_json(self, data):
        file_path = f"{self.save_root}/{self.ann_file}"
        dir_name, file_name = get_dir_and_file_name(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(data, file)
            print(f"save json file: {file_path}")
        return True

    def save_images(self, data):
        for image in data['images']:
            save_path = f"{self.save_root}/{self.img_prefix}"
            os.makedirs(save_path, exist_ok=True)
            os.system(f"cp {self.data_root}/{self.img_prefix}/{image['file_name']} {save_path}/{image['file_name']}")
            # print(f"{self.data_root}/{self.img_prefix}/{image['file_name']} --> {save_path}/{image['file_name']}")
        print(f"save images: {save_path}")
        return True


def main():
    '''
    Examples:
        python get_mini_dataset.py \
            --data_root /ws/data/coco \
            --ann_file annotations/instances_train2017.json \
            --img_prefix train2017 \
            --save_root /ws/data/coco_mini10 \
            --num_images 10
    '''
    parser = argparse.ArgumentParser(description='Get mini dataset')
    # From
    parser.add_argument('--data_root', type=str, default='/ws/data/coco', help='Data root')
    parser.add_argument('--ann_file', type=str, default='annotations/instances_train2017.json', help='Annotation file')
    parser.add_argument('--img_prefix', type=str, default='train2017', help='Image prefix from')
    # To
    parser.add_argument('--save_root', type=str, default='/ws/data/coco_mini10', help='Save root')
    # Setting
    parser.add_argument('--num_images', type=int, default=10, help='Number of images')

    args = parser.parse_args()

    mini_dataset = MiniCOCODataset(args.data_root, args.ann_file, args.save_root, args.img_prefix)
    mini_data = mini_dataset.get_mini_dataset(num_images=args.num_images)
    mini_dataset.save_json(mini_data)
    mini_dataset.save_images(mini_data)


if __name__ == '__main__':
    main()
