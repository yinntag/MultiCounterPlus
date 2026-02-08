import json
import time
import torch

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
periodicity_threshold = 0.3

input_files = [
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_reps_stride1.json',
    '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_reps_stride2.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_reps_stride3.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_reps_stride4.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_reps_stride5.json',

]

output_files = [
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_converted1.json',
    '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_converted2.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_converted3.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_converted4.json',
    # '/mnt/tbdisk/tangyin/MultiCounter/demo_video/intermediate_results/results_periods_converted5.json',
]


def period_segment(period_count):
    repeats_int = int(period_count[-1].item())
    period_segments = []
    for i in range(repeats_int):
        start_frames = torch.nonzero(torch.abs(period_count - i) == torch.abs(period_count - i).min()).max().item()
        end_frame = torch.argmin(torch.abs(period_count - (i + 1))).item()
        period_segments.append([start_frames, end_frame])
    return period_segments


def process_periods(raw_period_lengths):
    # Convert raw_period_lengths to a tensor
    raw_period_lengths = torch.tensor(raw_period_lengths)

    period_length_confidence, per_frame_period_length = torch.max(torch.softmax(raw_period_lengths, dim=-1), dim=-1)

    per_frame_period_length = (per_frame_period_length + 1)
    per_frame_period_length = [1.0 / i if i != 0 and i > 1 else 0 for i in per_frame_period_length]

    tensor_list = [torch.tensor([item], dtype=torch.float32) for item in per_frame_period_length]
    cumulative_tensor = torch.cat(tensor_list)

    cumulative_sum = torch.cumsum(cumulative_tensor, dim=0)

    per_frame_counts_cs = period_segment(cumulative_sum)

    per_frame_counts_cs = [[x * 1 for x in interval] for interval in per_frame_counts_cs]

    return per_frame_counts_cs


for input_file, output_file in zip(input_files, output_files):
    print(f'Processing {input_file}...')

    results = json.load(open(input_file, 'r'))
    filtered_results = []

    for query in results:
        periods_converted = []
        raw_period_lengths = query['period_scores']
        periods_converted = process_periods(raw_period_lengths)
        query.update({'periods_converted': periods_converted})
        filtered_results.append(query)

    with open(output_file, 'w') as f:
        json.dump(filtered_results, f, indent=2)

print('Done')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
