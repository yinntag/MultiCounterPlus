### Installation

1. Create a new conda environment:

   ```bash
   conda create -n multicounter python=3.9
   conda activate multicounter
   ```
   
2. Install Pytorch (1.7.1 is recommended), scipy, tqdm, pandas.

3. Install MMDetection. 

   * Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) first. 1.4.8 is recommended.

   * ```bash
     cd MultiCounter
     pip install -v -e .
     ```

### Demo

You can put some videos in `demo_video/source_video/` and get the visualization inference result in `demo_video/visual_result/` by running the following command:

  ```bash
  bash tools/code_for_demo/demo.sh
  ```

### Train
You can train your own model by running the following command:

```bash
bash tools/train.sh

```

### Test
You can test your own model by running the following command:

```bash
bash tools/test.sh
```
