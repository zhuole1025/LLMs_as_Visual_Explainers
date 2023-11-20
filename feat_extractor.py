import os
import json
import shutil
import argparse
from tqdm import tqdm

import clip
import torch
from PIL import Image

def parser_args():
    parser = argparse.ArgumentParser(description='preprocessing parameters')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--split_dir', type=str, help='split json file directory')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='backbone model')
    return parser.parse_args()

def main():
    args = parser_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device)
    with open(args.split_dir, 'r') as json_file:
        data = json.load(json_file)

    categories = []
    for split in ['train', 'val', 'test']:
        print('Processing {} data, containing {} images'.format(split, len(data[split])))
        for item in data[split]:
            file_path, subfolder, category = item
            if category not in categories:
                categories.append(category)
            old_file_path = os.path.join(args.data_dir, file_path)
            new_folder_path = os.path.join(args.output_dir, 'tmp', split, str(subfolder))
            os.makedirs(new_folder_path, exist_ok=True)
            if len(file_path.split('/')) > 1:
                new_file_path = os.path.join(new_folder_path, file_path.split('/')[-1])
            else:
                new_file_path = os.path.join(new_folder_path, file_path)
            shutil.copy(old_file_path, new_file_path)

        source_folder = os.path.join(args.output_dir, 'tmp', split)
        target_folder = os.path.join(args.output_dir, split)
        os.makedirs(target_folder, exist_ok=True)
        for class_folder in tqdm(os.listdir(source_folder)):
            class_folder_path = os.path.join(source_folder, class_folder)
            if os.path.isdir(class_folder_path) and class_folder_path.split('/')[-1].isdigit():
                image_features = []
                for image_path in os.listdir(class_folder_path):
                    if image_path.endswith('.jpg') or image_path.endswith('.JPEG'):
                        image = preprocess(Image.open(os.path.join(class_folder_path, image_path))).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features.append(model.encode_image(image))
                image_features = torch.stack(image_features, dim=0)
                target_filepath = os.path.join(target_folder, f"{class_folder}.pt")
                torch.save(image_features, target_filepath)

    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        f.write(str(categories))
    
    shutil.rmtree(os.path.join(args.output_dir, 'tmp'))
    print("Done!")

if __name__ == '__main__':
    main()