import json
import os
from argparse import ArgumentParser
from threading import Thread
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import collate, scatter
from tqdm import tqdm
import math
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, flops_to_string, params_to_string
from mmdet.core import build_assigner
from mmcv import ConfigDict
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/instcount/instcount_r50.py',
                        help='test config file path')
    parser.add_argument('--checkpoint', default='./log/latest.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--json',
        default="./MRepData/annotations/test.json",
        help='Path to multirep test json file')
    parser.add_argument(
        '--root', default="./MRepData/test_raw_frames/", help='Path to image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))


def compute_iou(assigner, previous, cur):
    iou = assigner.assign(previous, cur)
    return iou


def main(args):
    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)  # build_detector
    model = add_flops_counting_methods(model)
    cfg = model.cfg
    anno = json.load(open(args.json))
    test_pipeline = Compose(cfg.data.test.pipeline)

    results = []
    clip_len = 64  # define the video clip length for a single forward propagation
    stride = 32  # define the stride
    matcher_config = ConfigDict(dict(type='InstLinkerCalculator',
                                     cls_cost=dict(type='FocalLossCost', weight=2.0),
                                     reg_cost=dict(type='InferenceBBoxL1Cost', weight=5.0),
                                     iou_cost=dict(type='IoUCost', iou_mode='iou',
                                                   weight=1.0)))
    assigner = build_assigner(matcher_config)  # perform Hungarian matching
    iou_threshold = 0.2
    person_threshold = 0.5
    for video in tqdm(anno['videos']):
        imgs = video['file_names']
        video_det_bboxes = []
        video_det_periodicity = []
        video_det_periods = []

        datas, threads = [], []
        video_length = len(imgs)

        if video_length <= clip_len:
            clip_num = 1
        else:
            clip_num = math.ceil((video_length - clip_len) / stride) + 1
        for clip_index in range(0, clip_num):
            if clip_index != clip_num - 1:  # Determine if it is the last clip
                cur_clip = imgs[clip_index * stride:clip_index * stride + clip_len]
                clip_overlap = clip_len - stride
            else:  # If it is the last clip, take the last clip_num frame backwards
                cur_clip = imgs[-clip_len:]
                if (video_length - clip_len) % stride:
                    clip_overlap = clip_len - (video_length - clip_len) % stride
                else:
                    clip_overlap = clip_len - stride
            threads = []
            datas = []
            for img in cur_clip:
                data = dict(img_info=dict(filename=img), img_prefix=args.root)
                threads.append(Thread(target=load_datas, args=(data, test_pipeline, datas)))
                threads[-1].start()
            for thread in threads:
                thread.join()

            datas = sorted(datas, key=lambda x: x['img_metas'].data['filename'])
            datas = collate(datas, samples_per_gpu=len(cur_clip))  # form the input batch
            datas['img_metas'] = datas['img_metas'].data
            datas['img'] = datas['img'].data
            datas = scatter(datas, [args.device])[0]
            # print(datas)
            # print(model)
            with torch.no_grad():
                model.start_flops_count()
                (det_bboxes, det_labels), det_periods, det_periodicity, tsm = model(
                    return_loss=False,
                    rescale=True,
                    format=False,
                    **datas)  # det_bboxes: [x1,y1,x2,y2].
                # print("det_periodicity:", det_periodicity)
                # print("det_periods:", det_periods)
                # sys.exit()
                # _, params_count = model.compute_average_flops_cost()
                # print(params_to_string(params_count))
                model.stop_flops_count()

            # Perform inter-clip matching
            if clip_index != 0:
                previous_det_bboxes_for_match = video_det_bboxes[:, -clip_overlap:, :]

                det_bboxes = torch.stack(det_bboxes)
                det_periodicity = torch.stack(det_periodicity)
                det_periods = torch.stack(det_periods)

                det_bboxes = det_bboxes.permute(1, 0, 2)
                det_periodicity = det_periodicity.permute(1, 0, 2)
                det_periods = det_periods.permute(1, 0, 2)

                # filter prediction results by a confidence threshold
                det_periodicity = det_periodicity[torch.where(det_bboxes[:, 0, -1] > person_threshold)]
                det_bboxes = det_bboxes[torch.where(det_bboxes[:, 0, -1] > person_threshold)]
                det_periods = det_periods[torch.where(det_bboxes[:, 0, -1] > person_threshold)]

                previous_person_num = previous_det_bboxes_for_match.size(0)

                next_padding_bboxes = torch.zeros([previous_person_num, clip_len - clip_overlap, 5]).to(
                    video_det_bboxes.device)
                video_det_bboxes = torch.cat((video_det_bboxes, next_padding_bboxes), 1)

                next_padding_periodicity = torch.zeros([previous_person_num, clip_len - clip_overlap, 1]).to(
                    video_det_periodicity.device)
                video_det_periodicity = torch.cat((video_det_periodicity, next_padding_periodicity), 1)

                next_padding_periods = torch.zeros([previous_person_num, clip_len - clip_overlap, 32]).to(
                    video_det_periods.device)
                video_det_periods = torch.cat((video_det_periods, next_padding_periods), 1)

                # perform matching
                previous_det_bboxes_for_iou = previous_det_bboxes_for_match.permute(1, 0, 2)
                det_boxes_for_iou = det_bboxes.permute(1, 0, 2)[:clip_overlap, :, :]
                mat = assigner.assign(previous_det_bboxes_for_iou, det_boxes_for_iou, datas['img_metas'][0][0])

                det_assigned = torch.zeros(det_bboxes.shape[0])
                for i in range(0, min(mat.shape)):

                    tar = np.unravel_index(mat.argmax(), mat.shape)
                    if mat[tar[0], tar[
                        1]] < iou_threshold:  # below the threshold value, indicating the appearance of new id, added to the registry
                        new_person_bboxes = det_bboxes[tar[1], -(clip_len):, :].unsqueeze(0)
                        new_person_periodicity = det_periodicity[tar[1], -(clip_len):, :].unsqueeze(0)
                        new_person_periods = det_periods[tar[1], -(clip_len):, :].unsqueeze(0)

                        new_person_pre_bboxes = torch.zeros([1, video_det_bboxes.size(1) - (clip_len), 5]).to(
                            video_det_bboxes.device)
                        new_person_pre_periodicity = torch.zeros([1, video_det_periodicity.size(1) - (clip_len), 1]).to(
                            video_det_periodicity.device)
                        new_person_pre_periods = torch.zeros([1, video_det_periods.size(1) - (clip_len), 32]).to(
                            video_det_periods.device)

                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_periodicity = torch.cat((new_person_pre_periodicity, new_person_periodicity), 1)
                        new_person_periods = torch.cat((new_person_pre_periods, new_person_periods), 1)

                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_periodicity = torch.cat((video_det_periodicity, new_person_periodicity), 0)
                        video_det_periods = torch.cat((video_det_periods, new_person_periods), 0)

                        mat[tar[0], :] = -10000
                        mat[:, tar[1]] = -10000
                        det_assigned[tar[1]] = 1  # Mark the new prediction result for index = tar[1] has been processed
                    else:  # the current match is satisfying the threshold
                        mat[tar[0], :] = -10000
                        mat[:, tar[1]] = -10000

                        video_det_bboxes[tar[0], -(clip_len - clip_overlap):, :] = det_bboxes[tar[1],
                                                                                   -(clip_len - clip_overlap):, :]
                        video_det_periodicity[tar[0], -(clip_len - clip_overlap):, :] = det_periodicity[tar[1],
                                                                                   -(clip_len - clip_overlap):, :]
                        video_det_periods[tar[0], -(clip_len - clip_overlap):, :] = det_periods[tar[1],
                                                                                   -(clip_len - clip_overlap):, :]

                        # Average the result on overlapping parts
                        video_det_bboxes[tar[0], -clip_len:-(clip_len - clip_overlap), :] = (video_det_bboxes[tar[0],
                                                                                             -clip_len:-(
                                                                                                         clip_len - clip_overlap),
                                                                                             :] + det_bboxes[tar[1],
                                                                                                  -clip_len:-(
                                                                                                              clip_len - clip_overlap),
                                                                                                  :]) / 2
                        video_det_periodicity[tar[0], -clip_len:-(clip_len - clip_overlap), :] = (video_det_periodicity[tar[0],
                                                                                             -clip_len:-(
                                                                                                         clip_len - clip_overlap),
                                                                                             :] + det_periodicity[tar[1],
                                                                                                  -clip_len:-(
                                                                                                              clip_len - clip_overlap),
                                                                                                  :]) / 2
                        video_det_periods[tar[0], -clip_len:-(clip_len - clip_overlap), :] = (video_det_periods[tar[0],
                                                                                             -clip_len:-(
                                                                                                     clip_len - clip_overlap),
                                                                                             :] + det_periods[tar[1],
                                                                                                  -clip_len:-(
                                                                                                          clip_len - clip_overlap),
                                                                                                  :]) / 2

                        det_assigned[tar[1]] = 1  # Mark the new prediction result for index = tar[1] has been processed
                for index in range(0, det_assigned.shape[0]):
                    if det_assigned[
                        index] == 0:  # This new prediction result has not been processed yet and is a new id

                        new_person_bboxes = det_bboxes[index, -(clip_len):, :].unsqueeze(0)
                        new_person_periodicity = det_periodicity[index, -(clip_len):, :].unsqueeze(0)
                        new_person_periods = det_periods[index, -(clip_len):, :].unsqueeze(0)

                        new_person_pre_bboxes = torch.zeros([1, video_det_bboxes.size(1) - (clip_len), 5]).to(
                            video_det_bboxes.device)
                        new_person_pre_periodicity = torch.zeros([1, video_det_periodicity.size(1) - (clip_len), 1]).to(
                            video_det_periodicity.device)
                        new_person_pre_periods = torch.zeros([1, video_det_periods.size(1) - (clip_len), 32]).to(
                            video_det_periods.device)

                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_periodicity = torch.cat((new_person_pre_periodicity, new_person_periodicity), 1)
                        new_person_periods = torch.cat((new_person_pre_periods, new_person_periods), 1)

                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_periodicity = torch.cat((video_det_periodicity, new_person_periodicity), 0)
                        video_det_periods = torch.cat((video_det_periods, new_person_periods), 0)

                        det_assigned[index] = 1  # Mark the new prediction result for index = tar[1] has been processed

            else:  # for the first video_cilp

                det_bboxes = torch.stack(det_bboxes)
                det_periodicity = torch.stack(det_periodicity)
                det_periods = torch.stack(det_periods)

                det_bboxes = det_bboxes.permute(1, 0, 2)
                det_periodicity = det_periodicity.permute(1, 0, 2)
                det_periods = det_periods.permute(1, 0, 2)

                video_det_periodicity = det_periodicity[torch.where(det_bboxes[:, 0, -1] > person_threshold)]
                video_det_bboxes = det_bboxes[torch.where(det_bboxes[:, 0, -1] > person_threshold)]
                video_det_periods = det_periods[torch.where(det_bboxes[:, 0, -1] > person_threshold)]

        det_bboxes = video_det_bboxes.permute(1, 0, 2)  # [total_frames, gt_num_object, 5] 4 for bboxes, 1 for scores
        det_periodicity = video_det_periodicity.permute(1, 0, 2)  # [total_frames, gt_num_object, 1] 1 for periodicity scores
        det_periods = video_det_periods.permute(1, 0, 2)  # [total_frames, gt_num_object, 64] 64 for period scores
        print('\n')
        print('final_det_bboxes:', det_bboxes.shape)
        print('final_det_periodicity:', det_periodicity.shape)
        print('final_det_periods:', det_periods.shape)

        for inst_ind in range(det_bboxes.size(1)):  # get basic info like video_id, score, category_id(person or not)
            objs = dict(
                video_id=video['id'],
                score=det_bboxes[:, inst_ind, -1][torch.where(det_bboxes[:, inst_ind, -1] > 0)].mean().item(),
                category_id=1,
                bboxes=[],
                periodicity_scores=[],
                period_scores=[],
                score_per_img=[]
            )
            for sub_ind in range(det_bboxes.size(0)):  # each frame results like bboxes, periodicity_scores, score_per_img
                m = det_bboxes[
                    sub_ind, inst_ind, :-1].detach().cpu().numpy().tolist()
                n = det_periods[
                    sub_ind, inst_ind].detach().cpu().numpy().tolist()
                if (m[0] + m[1] + m[2] + m[3]) == 0:
                    m = None
                else:
                    m = [m[0], m[1], m[2] - m[0], m[3] - m[1]]
                objs['bboxes'].append(m)
                objs['periodicity_scores'].append(det_periodicity[sub_ind, inst_ind].item())
                objs['period_scores'].append(n)
                objs['score_per_img'].append(det_bboxes[sub_ind, inst_ind, -1].item())
            results.append(objs)

    # export results to json format
    os.makedirs('results', exist_ok=True)
    write_path = os.path.join('results',
                              f'res0_{args.config.rstrip(".py").split("/")[-1]}_{args.json.split("/")[-1]}')
    json.dump(results, open(write_path, 'w'), indent=2)
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    args = parse_args()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    main(args)
