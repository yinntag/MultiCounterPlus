import os
import time
import cv2
import json
from tqdm import tqdm
from argparse import ArgumentParser

# generate raw frames and gt json files from video dataset
parser = ArgumentParser()

parser.add_argument('--root', default="./MRepData/", help='Path to dataset root')
args = parser.parse_args()

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

video_dataset_root = args.root

split_dirs = ['train', 'val', 'test']


def assign_period_lengths(total_frames, periods):
    period_lengths = [0] * total_frames
    for period in periods:
        start_frame, end_frame = period
        period_length = end_frame - start_frame

        for frame in range(start_frame, end_frame):
            period_lengths[frame] = period_length

    return period_lengths


for split_dir in split_dirs:
    split_dataset_root = os.path.join(video_dataset_root, split_dir)
    if not os.path.exists(split_dataset_root):
        continue
    rawframes_dataset_root = os.path.join(video_dataset_root, f'{split_dir}_raw_frames')
    video_list = os.listdir(split_dataset_root)
    video_list = list(map(str, video_list))
    video_list.sort()
    dataset = {}

    info = {'info': {'description': 'MultiRep Dataset', 'url': '1', 'version': '1', 'year': '2023',
                     'contributor': 'Yin Tang',
                     'data_created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}}
    licenses = {'licenses': 'only for research'}
    categories = {'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person'}]}

    videos = []
    annotations = []
    anno_id = 1

    target_width = 224
    target_height = 224
    for video_sample in tqdm(video_list):
        video_path = os.path.join(split_dataset_root, str(video_sample), video_sample + '.mp4')
        anno_path = os.path.join(split_dataset_root, str(video_sample), video_sample + '.json')
        origin_anno = json.load(open(anno_path, 'r'))

        height = origin_anno.pop('height')
        width = origin_anno.pop('width')
        length = origin_anno.pop('length')
        video_name = origin_anno.pop('video_name')
        scale_w = target_width / width
        scale_h = target_height / height

        # record video-related information and store the individual frames
        video = {'height': target_height, 'width': target_width, 'length': length}
        file_names = []
        camera = cv2.VideoCapture(video_path)
        save_dir = os.path.join(rawframes_dataset_root, str(video_sample))
        os.makedirs(save_dir, exist_ok=True)
        img_index = 0
        while True:
            res, image = camera.read()
            if not res:
                break
            relative_img_path = str(video_sample) + '/' + str(img_index).rjust(5, '0') + '.png'
            image = cv2.resize(image, (target_width, target_height))
            cv2.imwrite(os.path.join(rawframes_dataset_root, relative_img_path), image)
            file_names.append(relative_img_path)
            img_index += 1
        camera.release()
        video.update({'file_names': file_names})
        video.update({'id': video_sample})
        videos.append(video)

        # record annotation-related information
        for person in origin_anno:
            anno = {'height': target_height, 'width': target_width, 'length': 1, 'category_id': 1}
            anno_periodicity = []
            # resize the gt annotation according to the shape of resized frames (640*360 in our experiment)
            for index in range(0, length):
                if origin_anno[person]['bbox'][index] == None:
                    anno_periodicity.append(None)
                    continue
                else:
                    origin_anno[person]['bbox'][index][0] = origin_anno[person]['bbox'][index][0] * scale_w
                    origin_anno[person]['bbox'][index][1] = origin_anno[person]['bbox'][index][1] * scale_h
                    new_x1 = origin_anno[person]['bbox'][index][0]
                    new_y1 = origin_anno[person]['bbox'][index][1]
                    origin_anno[person]['bbox'][index][2] = origin_anno[person]['bbox'][index][2] * scale_w - new_x1
                    origin_anno[person]['bbox'][index][3] = origin_anno[person]['bbox'][index][3] * scale_h - new_y1
                periodicity_sign = 0
                for periodicity_index in range(0, len(origin_anno[person]['period'])):
                    if index >= origin_anno[person]['period'][periodicity_index][0] and index <= \
                            origin_anno[person]['period'][periodicity_index][1]:
                        periodicity_sign = 1
                        break
                anno_periodicity.append(periodicity_sign)
            anno.update({'bboxes': origin_anno[person]['bbox']})
            anno.update({'periodicity': origin_anno[person]['period']})
            anno.update({'periodicity_binary': anno_periodicity})
            anno.update({'periods': assign_period_lengths(length, origin_anno[person]['period'])})
            anno.update({'video_id': video_sample})
            anno.update({'id': anno_id})

            anno_id += 1

            annotations.append(anno)

    dataset.update(info)
    dataset.update(licenses)
    dataset.update({'videos': videos})
    dataset.update(categories)
    dataset.update({'annotations': annotations})

    final_json_root = os.path.join(video_dataset_root, 'annotations')
    os.makedirs(final_json_root, exist_ok=True)
    json.dump(dataset, open(os.path.join(final_json_root, f'{split_dir}.json'), 'w'), indent=2)
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
