import os
import ast
import json
import random
import pickle
import argparse
import numpy as np

from utils import *


def parser_args():
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--img_dir', type=str, help='path to image directory')
    parser.add_argument('--label_dir', type=str, help='path to label text')
    parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
    parser.add_argument('--baseline', type=str, default='', help='baseline method')
    parser.add_argument('--before_text', type=str, default='', help='text before class name')
    parser.add_argument('--between_text', type=str, default=', ', help='text between class name and descriptor')
    parser.add_argument('--after_text', type=str, default='', help='text after descriptor')
    parser.add_argument('--apply_descriptor_modification', type=bool, default=True, help='whether to apply descriptor modification')
    parser.add_argument('--model_size', type=str, default='ViT-B/32', help='model size')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parser_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = torch.half if args.device == "cuda" else torch.float32
    model, preprocess = clip.load(args.model_size, device=torch.device("cpu"))
    model = model.to(args.device)

    # load image embeddings
    test_image_embs, test_gt = load_image_embeddings(os.path.join(args.img_dir, 'test'), -1)
    test_image_embs = test_image_embs / test_image_embs.norm(dim=1, keepdim=True)
    test_image_embs = test_image_embs.to(args.device).to(dtype)

    # read label and prompt text
    with open(args.label_dir, 'r') as f:
        if args.label_dir.endswith('.json'):
            init_full_class_names = json.load(f)
        else:
            content = f.read()
            init_full_class_names = ast.literal_eval(content)

    # evaluate the performance on the test set
    eval_metrics = evaluate(init_full_class_names, test_image_embs, test_gt, model, args.device, baseline=args.baseline,
                            before_text=args.before_text, between_text=args.between_text, after_text=args.after_text,
                            apply_descriptor_modification=args.apply_descriptor_modification)
    
    # print and save results
    print(f"Test Overall Accuracy: {eval_metrics['all_acc']:.2%}")
    with open(os.path.join(args.output_dir, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(eval_metrics, f)


if __name__ == '__main__':
    main()
