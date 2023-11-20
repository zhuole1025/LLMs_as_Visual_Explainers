import argparse
import ast
import time
import json
import random
import re
import logging

import numpy as np
import openai
from tqdm import tqdm

from utils import *

def parser_args():
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--img_dir', type=str, help='path to image directory')
    parser.add_argument('--label_dir', type=str, help='path to label text')
    parser.add_argument('--prompt_dir', type=str, default='prompt', help='path to prompt text')
    parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
    parser.add_argument('--n_iters', type=int, default=10, help='number of iterations for optimization')
    parser.add_argument('--n_shots', type=int, default=-1, help='number of shots for optimization')
    parser.add_argument('--n_samples', type=int, default=4, help='number of samples for optimization')
    parser.add_argument('--n_desc_init', type=int, default=30, help='number of descriptions to initialize')
    parser.add_argument('--n_desc', type=int, default=15, help='number of descriptions to optimize each time')
    parser.add_argument('--cluster_size', type=int, default=10, help='number of classes in each cluster')
    parser.add_argument('--visual_type', type=str, default='confusion_thresh', help='type of visual feedback')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for LLM sampling')
    parser.add_argument('--gpt', type=str, default='gpt-4-1106-preview', help='gpt model version')
    parser.add_argument('--baseline', type=str, default='', help='baseline method')
    parser.add_argument('--before_text', type=str, default='', help='text before class name')
    parser.add_argument('--between_text', type=str, default=', ', help='text between class name and descriptor')
    parser.add_argument('--after_text', type=str, default='', help='text after descriptor')
    parser.add_argument('--apply_descriptor_modification', type=bool, default=True, help='whether to apply descriptor modification')
    parser.add_argument('--model_size', type=str, default='ViT-B/32', help='model size')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parser_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO,
        filename=f'{args.output_dir}/{time.asctime()}.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
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
    train_image_embs, train_gt = load_image_embeddings(os.path.join(args.img_dir, 'train'), args.n_shots)
    test_image_embs, test_gt = load_image_embeddings(os.path.join(args.img_dir, 'test'), -1)
    train_image_embs = train_image_embs / train_image_embs.norm(dim=1, keepdim=True)
    test_image_embs = test_image_embs / test_image_embs.norm(dim=1, keepdim=True)
    train_image_embs = train_image_embs.to(args.device).to(dtype)
    test_image_embs = test_image_embs.to(args.device).to(dtype)

    # read label and prompt text
    with open(args.label_dir, 'r') as f:
        if args.label_dir.endswith('.json'):
            init_full_class_names = json.load(f)
        else:
            content = f.read()
            init_full_class_names = ast.literal_eval(content)
    with open(os.path.join(args.prompt_dir, 'init_system.txt'), 'r') as f:
        init_system = f.read()
    with open(os.path.join(args.prompt_dir, 'init_user.txt'), 'r') as f:
        init_user = f.read()
    with open(os.path.join(args.prompt_dir, 'mutation_system.txt'), 'r') as f:
        init_mutation_system = f.read()
    with open(os.path.join(args.prompt_dir, 'mutation_user.txt'), 'r') as f:
        init_mutation_user = f.read()
    with open(os.path.join(args.prompt_dir, 'crossover_system.txt'), 'r') as f:
        init_crossover_system = f.read()
    with open(os.path.join(args.prompt_dir, 'crossover_user.txt'), 'r') as f:
        init_crossover_user = f.read()

    all_prompt_tokens = 0
    all_completion_tokens = 0
    all_total_tokens = 0
    best_acc_overall = 0
    best_full_class_names = init_full_class_names.copy()
    cur_full_class_names = init_full_class_names.copy()
    positive_bank = [[] for _ in range(len(cur_full_class_names))]
    negative_bank = [[] for _ in range(len(cur_full_class_names))]
    for iter in tqdm(range(args.n_iters)):
        backup_full_class_names = cur_full_class_names.copy()
        cluster_idx, n_clusters = cluster_texts(cur_full_class_names, model, args.device, args.cluster_size)
        for cluster_id in range(n_clusters):
            class_idxs = torch.tensor([i for i in range(len(cur_full_class_names)) if cluster_idx[i] == cluster_id])
            class_names = [cur_full_class_names[i] for i in class_idxs]
            init_names = [init_full_class_names[i] for i in class_idxs]
            mask = (train_gt[:, None] == class_idxs).any(dim=1)
            cluster_gt = train_gt[mask]
            cluster_gt = (cluster_gt[:, None] == class_idxs).nonzero(as_tuple=True)[1]
            cluster_image_embs = train_image_embs[mask]
            metrics = evaluate(class_names, cluster_image_embs, cluster_gt, model, args.device)
            visual_feedback = build_visual_feedback(args.visual_type, metrics, init_names)
            
            # build input prompt for mutation
            if iter == 0:
                user_prompt = init_user.format(current_class=list2str(class_names), visual_feedback=visual_feedback)
                system_prompt = init_system.format(n_desc_init=args.n_desc_init)
            else:
                user_prompt = init_mutation_user.format(current_class=list2str(class_names), visual_feedback=visual_feedback,
                                                        positive_list=build_memory_bank(positive_bank, init_full_class_names, class_idxs),
                                                        negative_list=build_memory_bank(negative_bank, init_full_class_names, class_idxs))
                system_prompt = init_mutation_system.format(n_desc=args.n_desc)

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            # save input prompt
            with open(os.path.join(args.output_dir, f"iter_{iter}_mutation_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(system_prompt + '\n' + user_prompt)

            if args.debug:
                # create dummy response
                logging.info(system_prompt)
                logging.info(user_prompt)
                response = {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "choices": [
                                   {"message": {"content": str(class_names)}}
                               ] * args.n_samples
                }
            else:
                for attempt in range(100):
                    try:
                        # generate response
                        response = openai.ChatCompletion.create(
                            model=args.gpt,
                            messages=messages,
                            temperature=args.temperature,
                            n=args.n_samples,
                        )
                        break
                    except Exception as e:
                        logging.info(e)
                        logging.info(f"Failed attempt {attempt} for cluster {cluster_id}")
                        time.sleep(10)

            responses = response["choices"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            total_completion_token = response["usage"]["completion_tokens"]
            total_token = response["usage"]["total_tokens"]
            # save responses
            with open(os.path.join(args.output_dir, f"iter_{iter}_cluster_{cluster_id}_mutation_response.txt"), 'w', encoding='utf-8') as f:
                f.write(str(responses))

            valid_responses = []
            valid_feedback = []
            best_acc = 0
            best_index = 0
            for response_id in range(args.n_samples):
                response_cur = responses[response_id]["message"]["content"]

                # Regex patterns to extract python list enclosed in GPT response
                pattern = r"\[.*?\]"
                class_names = re.search(pattern, response_cur, re.DOTALL)
                try:
                    class_names = class_names.group().strip()
                    class_names = ast.literal_eval(class_names)
                    assert len(class_names) == len(class_idxs)
                except:
                    logging.info(
                        f"Iteration {iter}, Cluster {cluster_id}, Response {response_id}: No valid class labels found in response\n{response_cur}")
                    continue
                
                valid_responses.append(class_names)
                metrics = evaluate(class_names, cluster_image_embs, cluster_gt, model, args.device)
                valid_feedback.append(metrics)
                logging.info(
                    f"Iteration {iter}, Cluster {cluster_id}, Response {response_id}, Overall Accuracy: {metrics['all_acc']:.2%}")

                if metrics['all_acc'] > best_acc:
                    best_acc = metrics['all_acc']
                    best_index = len(valid_responses) - 1

            # repeat the iteration if no valid response is found
            if len(valid_responses) == 0:
                logging.info(f"Iteration {iter}, Cluster {cluster_id}: No valid response found")
                continue
            
            if args.n_samples > 1:
                # Build input prompt for crossover
                class_samples = '\n'.join([f"Version {i}:\n{list2str(class_names)}\nOverall Accuracy: {feedback['all_acc']:.2%}\nClass-wise Accuracy:\n{feedback['class_acc']}\n" for i, (class_names, feedback) in enumerate(zip(valid_responses, valid_feedback))])
                user_prompt = init_crossover_user.format(class_samples=class_samples, n_samples=len(valid_responses))
                system_prompt = init_crossover_system.format(n_samples=len(valid_responses), n_desc_init=args.n_desc_init)

                # save input prompt
                with open(os.path.join(args.output_dir, f"iter_{iter}_cluster_{cluster_id}_crossover_prompt.txt"), 'w', encoding='utf-8') as f:
                    f.write(system_prompt + '\n' + user_prompt)

                crossover_messages = [{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}]
                if args.debug:
                    logging.info(system_prompt)
                    logging.info(user_prompt)
                    # create dummy response
                    crossover_response = {
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "choices": [
                            {"message": {"content": str(class_names)}}
                        ]
                    }
                else:
                    for attempt in range(100):
                        try:
                            crossover_response = openai.ChatCompletion.create(
                                model=args.gpt,
                                messages=crossover_messages,
                                temperature=args.temperature,
                            )
                            break
                        except Exception as e:
                            logging.info(e)
                            logging.info(f"Failed attempt {attempt} for cluster {cluster_id}")
                            time.sleep(10)
                crossover_responses = crossover_response["choices"]
                prompt_tokens += crossover_response["usage"]["prompt_tokens"]
                total_completion_token += crossover_response["usage"]["completion_tokens"]
                total_token += crossover_response["usage"]["total_tokens"]

                # save responses
                with open(os.path.join(args.output_dir, f"iter_{iter}_cluster_{cluster_id}_crossover_response.txt"), 'w', encoding='utf-8') as f:
                    f.write(str(crossover_responses))

                # extract the new class labels from the response
                response_cur = crossover_responses[0]["message"]["content"]
                pattern = r"\[.*?\]"
                class_names = re.search(pattern, response_cur, re.DOTALL)
                try:
                    class_names = class_names.group().strip()
                    class_names = ast.literal_eval(class_names)
                    assert len(class_names) == len(class_idxs)
                except:
                    class_names = valid_responses[best_index]
                    logging.info(f"Iteration {iter}: No class labels found in crossover response {response_cur}")

                metrics = evaluate(class_names, cluster_image_embs, cluster_gt, model, args.device)
                logging.info(f"Iteration {iter}, Cluster {cluster_id}, Crossover Overall Accuracy: {metrics['all_acc']:.2%}")
        
            # update current class labels
            best_class_names = valid_responses[best_index]
            if args.n_samples > 1 and metrics['all_acc'] > best_acc:
                best_acc = metrics['all_acc']
                best_class_names = class_names
            logging.info(f"Iteration {iter}, Cluster {cluster_id}, Best Overall Accuracy: {best_acc.item():.2%}")

            idx = 0
            for i in range(len(cur_full_class_names)):
                if cluster_idx[i] == cluster_id:
                    cur_full_class_names[i] = best_class_names[idx]
                    idx += 1

            # logging token information
            all_prompt_tokens += prompt_tokens
            all_completion_tokens += total_completion_token
            all_total_tokens += total_token
            logging.info(f"Iteration {iter}, Cluster {cluster_id}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        # evaluate the performance on the training set using all class labels
        if iter > 0:
            backup_metrics = full_metrics
        
        full_metrics = evaluate(cur_full_class_names, train_image_embs, train_gt, model, args.device, metrics=['all_acc', 'logits'])
        eval_metrics = evaluate(cur_full_class_names, test_image_embs, test_gt, model, args.device, metrics=['all_acc'])
        if full_metrics['all_acc'] > best_acc_overall:
            best_acc_overall = full_metrics['all_acc']
            best_full_class_names = cur_full_class_names.copy()
        
        # save the current response
        with open(os.path.join(args.output_dir, f"iter_{iter}_response.txt"), 'w', encoding='utf-8') as f:
            f.write(str(cur_full_class_names))
            
        logging.info(f"Iteration {iter}, Train Overall Accuracy: {full_metrics['all_acc']:.2%}")
        logging.info(f"Iteration {iter}, Test  Overall Accuracy: {eval_metrics['all_acc']:.2%}")

        # Update the positive and negative bank
        if iter > 0:
            try: 
                cur_full_keywords = [x.split(':')[1].strip().split(', ') for x in cur_full_class_names]
                backup_full_keywords = [x.split(':')[1].strip().split(', ') for x in backup_full_class_names]
            except:
                continue
            cur_full_keywords = [x.split(':')[1].strip().split(', ') for x in cur_full_class_names]
            backup_full_keywords = [x.split(':')[1].strip().split(', ') for x in backup_full_class_names]

            # Pre-compute logits for all classes
            remove_full_class_names = backup_full_class_names.copy()
            for i in tqdm(range(len(cur_full_keywords))):
                same_keywords = set(cur_full_keywords[i]) & set(backup_full_keywords[i])
                remove_full_class_names[i] = backup_full_class_names[i].split(':')[0] + ': ' + ', '.join(same_keywords)
            remove_metrics = evaluate(remove_full_class_names, train_image_embs, train_gt, model, args.device, metrics=['logits'])

            for i in tqdm(range(len(cur_full_keywords))):
                add_keywords = set(cur_full_keywords[i]) - set(backup_full_keywords[i])
                remove_keywords = set(backup_full_keywords[i]) - set(cur_full_keywords[i])
                same_keywords = set(cur_full_keywords[i]) & set(backup_full_keywords[i])
                
                add_logits = backup_metrics['logits'].clone()
                add_logits[:, i] = full_metrics['logits'][:, i]
                add_metrics = evaluate(backup_full_class_names, train_image_embs, train_gt, model, args.device, metrics=['all_acc'], logits=add_logits)
                add_score = add_metrics['all_acc']

                remove_logits = backup_metrics['logits'].clone()
                remove_logits[:, i] = remove_metrics['logits'][:, i]
                remove_metrics = evaluate(backup_full_class_names, train_image_embs, train_gt, model, args.device, metrics=['all_acc'], logits=remove_logits)
                remove_score = remove_metrics['all_acc']

                backup_score = backup_metrics['all_acc']
                if remove_score < backup_score:
                    positive_bank[i] += [kw for kw in remove_keywords if kw not in positive_bank[i]]
                elif remove_score > backup_score:
                    negative_bank[i] += [kw for kw in remove_keywords if kw not in negative_bank[i]]
                if add_score > remove_score:
                    positive_bank[i] += [kw for kw in add_keywords if kw not in positive_bank[i]]
                elif add_score < remove_score:
                    negative_bank[i] += [kw for kw in add_keywords if kw not in negative_bank[i]]

                same_keywords = set(cur_full_keywords[i]) & set(backup_full_keywords[i])
                positive_bank[i] = [kw for kw in positive_bank[i] if kw not in same_keywords][-10:]
                negative_bank[i] = [kw for kw in negative_bank[i] if kw not in same_keywords][-10:]

    # evaluate the performance on the test set
    eval_metrics = evaluate(best_full_class_names, test_image_embs, test_gt, model, args.device, metrics=['all_acc'], baseline=args.baseline,
                            before_text=args.before_text, between_text=args.between_text, after_text=args.after_text,
                            apply_descriptor_modification=args.apply_descriptor_modification)
    print(f"Test Overall Accuracy: {eval_metrics['all_acc']:.2%}")

    logging.info(f"All Prompt Tokens: {all_prompt_tokens}, All Completion Tokens: {all_completion_tokens}, All Total Tokens: {all_total_tokens}")

    # save the best response
    with open(os.path.join(args.output_dir, f"best_response.txt"), 'w', encoding='utf-8') as f:
        f.write(str(best_full_class_names))


if __name__ == '__main__':
    main()
