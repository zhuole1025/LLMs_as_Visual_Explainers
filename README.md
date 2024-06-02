# LLMs as Visual Explainers: Advancing Image Classification with Evolving Visual Descriptions

## Overview
This repository contains code to replicate key experiments from our paper [LLMs as Visual Explainers: Advancing Image Classification with Evolving Visual Descriptions]().

## Installation

First install the dependencies.

```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Follow [DATASETS.md](./DATASETS.md) to prepate the datasets. After that, you can run the following commands to extract the image features of the datasets.

```
python feat_extractor.py --data_dir $DATA --split_dir $SPLIT --output_dir data/$DATASET_NAME
```

where `$DATA` is the path to the dataset, `$SPLIT` is the path to the split file, and `$DATASET_NAME` is the name of the dataset (e.g., `imagenet`, `caltech`, etc.). The extracted features will be saved to `data/$DATASET_NAME`.

You need to set up the Openai API key to use GPT-4. Please set the environment variable in your terminal:

```
export OPENAI_API_KEY= "YOUR_API_KEY"
```


## Usage

To generate the class descriptions for a dataset using our method, you can run the following command:

```
python main.py --img_dir data/eurosat --label_dir data/eurosat/labels.txt
```

You can change more parameters in the `main.py` file.

To evaluate the generated descriptions, you can run the following command:

```
python eval.py --img_dir data/eurosat --label_dir result/Ours/eurosat_sota.txt
```

To reproduce the results in the paper (including baselines), you can run the commands in the `replicate_key_results.sh` file. 

## Citation

```
@article{han2023llms,
  title={LLMs as Visual Explainers: Advancing Image Classification with Evolving Visual Descriptions},
  author={Han, Songhao and Zhuo, Le and Liao, Yue and Liu, Si},
  journal={arXiv preprint arXiv:2311.11904},
  year={2023}
}
```
