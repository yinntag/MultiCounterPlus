from argparse import ArgumentParser
from mmcv import DictAction
from mmdet.datasets.multirep_api import MultiRep
from mmdet.datasets.multirep_eval_api import MultiRepEval
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--json',
        default="./MRepData/annotations/",
        help='Path to annotation json file')
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


def main(args):
    strides = [3]
    for stride in strides:
        multirep = MultiRep(os.path.join(args.json, 'test_stride{}.json'.format(stride)))  # GT-test data ground truth
        multirep_dets = multirep.loadRes('./result/results_periods_converted_stride{}.json'.format(stride))  # DT-test data detection results
        vid_ids = multirep.getVidIds()
        for res_type in ['bbox']:
            iou_type = res_type
            multirep_eval = MultiRepEval(multirep, multirep_dets, iou_type)
            multirep_eval.params.vidIds = vid_ids
            multirep_eval.evaluate()
            multirep_eval.accumulate()
            multirep_eval.action_ap()
            multirep_eval.summarize()


if __name__ == '__main__':
    args = parse_args()
    main(args)