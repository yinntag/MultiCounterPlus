import os
import cv2
from tqdm import tqdm
import json


def visual_pred_all(data_root):
    strides = [1, 2, 3, 4]

    for stride in strides:
        print(f"Processing stride {stride}...")
        visualization_save_path = os.path.join(data_root, f'visual_result')
        os.makedirs(visualization_save_path, exist_ok=True)

        video_info_list = json.load(open(os.path.join(data_root, 'intermediate_results', 'info.json'), 'r'))
        periodicity_det = json.load(open(os.path.join(data_root, f'intermediate_results/results_periods_converted{stride}.json'), 'r'))

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = [
                 (0, 250, 154), (220, 20, 60), (0, 255, 255), (0, 191, 255), (0, 0, 255),
                 (173, 255, 47), (218, 112, 214)]
        color_periodicity = (0, 255, 255)

        for video_info in tqdm(video_info_list['videos']):
            cur_video_id = video_info['id']
            focused_det = [det for det in periodicity_det if det['video_id'] == cur_video_id]

            # 获取视频 FPS
            video_path = os.path.join(data_root, video_info['file_names'][0])
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_capture.release()

            f = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(visualization_save_path, f'demo_{stride}_{cur_video_id}.mp4'),
                f,
                30,
                (video_info['width'], video_info['height'])
            )

            frame_index = -1
            periodicity_count_list = [0] * len(focused_det)

            for index, img_path in enumerate(video_info['file_names']):
                if index % stride == 0:  # 根据 stride 动态调整采样间隔
                    img = cv2.imread(os.path.join(data_root, 'intermediate_results', img_path))
                    person_index = 0
                    frame_index += 1

                    for person in focused_det:
                        if not person['bboxes']:
                            print(f"Warning: No more bounding boxes for person {person_index} at frame {frame_index}")
                            person_index += 1
                            continue
                        bbox = person['bboxes'].pop(0)
                        if bbox is None:
                            person_index += 1
                            continue

                        draw_color = color[person_index]
                        for periodicity_event in person['periods_converted']:
                            if frame_index >= periodicity_event[0] and frame_index <= periodicity_event[1]:
                                draw_color = color_periodicity
                                if frame_index == periodicity_event[1]:
                                    periodicity_count_list[person_index] += 1
                                break

                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                                      draw_color, 3)
                        cv2.putText(img, f'Obj{person_index}:  Rep{periodicity_count_list[person_index]}',
                                    (int(bbox[0]), max(0, int(bbox[1]) - 10)), font, 0.7, [0, 0, 255], 2)
                        person_index += 1

                    videoWriter.write(img)
            videoWriter.release()


if __name__ == '__main__':
    data_root = "/mnt/tbdisk/tangyin/MultiCounter/demo_video"
    visual_pred_all(data_root)
