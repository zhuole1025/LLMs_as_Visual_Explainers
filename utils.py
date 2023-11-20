import os
import clip
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from k_means_constrained import KMeansConstrained


def wordify(word):
    return word.replace('_', ' ')

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"

def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor
    
def load_image_embeddings(img_dir, n_shots):
    image_embs = []
    gt = []
    for file in os.listdir(img_dir):
        embs = torch.load(os.path.join(img_dir, file), map_location="cpu").squeeze(1)
        image_embs.append(embs)
        gt.extend([int(file.split(".")[0])] * embs.shape[0])
    gt = torch.tensor(gt)
    gt, idx = gt.sort()
    image_embs = torch.cat(image_embs, dim=0)
    image_embs = image_embs[idx]
    
    if n_shots > 0:
        n_classes = len(torch.unique(gt))
        sampled_indices = []
        for class_id in range(n_classes):
            class_indices = (gt == class_id).nonzero(as_tuple=True)[0]
            # If there are fewer samples than n_shots, take all available samples for this class
            num_samples_to_take = min(n_shots, class_indices.shape[0])
            sampled_indices_for_class = class_indices[torch.randperm(class_indices.shape[0])[:num_samples_to_take]]
            sampled_indices.append(sampled_indices_for_class)
        sampled_indices = torch.cat(sampled_indices)
        image_embs = image_embs[sampled_indices]
        gt = gt[sampled_indices]
    
    return image_embs, gt

def encode_text(model, text, device):
    text_tokens = clip.tokenize(text).to(device)
    text_embs = model.encode_text(text_tokens)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)
    return text_embs

def cluster_texts(class_names, model, device, cluster_size, size_max_factor=1.2):
    n_clusters = max(1, round(len(class_names) / cluster_size))
    class_names_enemble = build_class_enemble(class_names)
    text_features = []
    for i in range(len(class_names_enemble)):
        text_feature_i = encode_text(model, class_names_enemble[i], device).detach().cpu().numpy().mean(axis=0)
        text_features.append(text_feature_i)
    text_features = np.stack(text_features, axis=0)
    size_max = min(max(cluster_size * size_max_factor, len(class_names) / n_clusters), len(class_names))
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_max=size_max,
        n_jobs=-1
    )
    cluster_idx = kmeans.fit_predict(text_features)
    return cluster_idx, n_clusters

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

def evaluate(class_names, image_embs, gt, model, device, mode='mean', logits=None,
             metrics=['all_acc', 'class_acc', 'confusion_matrix', 'confusion_thresh', 'confusion_topk'],
             baseline='', before_text='', between_text=', ', after_text='', apply_descriptor_modification=False):
    if logits == None:
        class_names_enemble = build_class_enemble(class_names, baseline, before_text, between_text, after_text, apply_descriptor_modification)
        logits_per_image = []
        for i in range(len(class_names_enemble)):
            text_embs = encode_text(model, class_names_enemble[i], device).to(image_embs.dtype)
            logits = model.logit_scale * image_embs @ text_embs.t()
            logits = logits.detach().cpu().float()
            logits_per_image.append(aggregate_similarity(logits, mode))
        logits_per_image = torch.stack(logits_per_image, dim=1)
    else:
        logits_per_image = logits

    metrics = {
        'logits': None,
        'all_acc': None,
        'class_acc': None,
        'confusion_matrix': None,
        'confusion_thresh': None,
        'confusion_topk': None
    }
    if 'logits' in metrics:
        metrics['logits'] = logits_per_image
    if 'all_acc' in metrics:
        probs = logits_per_image.softmax(dim=-1)
        index = probs.argmax(dim=-1)
        all_acc = (index == gt).float().mean()
        metrics['all_acc'] = all_acc
    if 'class_acc' in metrics:
        class_acc = []
        for i in range(len(class_names)):
            idx = (gt == i)
            acc = (index[idx] == gt[idx]).float().mean().item() * 100
            acc = f"{acc:.2f}%"
            class_acc.append(acc)
        metrics['class_acc'] = class_acc
    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(gt, index)
        metrics['confusion_matrix'] = cm
    if 'confusion_thresh' in metrics or 'confusion_topk' in metrics:
        probs = logits_per_image.softmax(dim=-1).cpu()
        k = min(3, len(class_names) - 1)
        m = min(5, len(class_names) - 1)
    if 'confusion_thresh' in metrics:
        # Compute classes that might be confused based on the threshold
        gt_score = probs.gather(1, gt.unsqueeze(-1)).squeeze()
        thresholds = gt_score * 0.9
        above_threshold_matrix = probs >= thresholds.unsqueeze(-1)
        one_hot_gt = torch.eye(len(class_names))[gt]
        aggregated_thresh = one_hot_gt.t() @ above_threshold_matrix.float()
        aggregated_thresh.fill_diagonal_(-float('inf'))
        top_m_values, top_m_indices = aggregated_thresh.topk(m, dim=1)
        confusion_thresh = [(indices[values > 0]).tolist() for values, indices in zip(top_m_values, top_m_indices)]
        metrics['confusion_thresh'] = confusion_thresh
    if 'confusion_topk' in metrics:
        # Compute classes that might be confused based on the top-k
        top_k_indices_per_sample = probs.topk(k, dim=-1).indices
        one_hot_top_k = torch.zeros(probs.shape).scatter_(1, top_k_indices_per_sample, 1)
        one_hot_gt = torch.eye(len(class_names))[gt]
        aggregated_top_k = one_hot_gt.t() @ one_hot_top_k
        aggregated_top_k.fill_diagonal_(-float('inf'))
        top_m_values, top_m_indices = aggregated_top_k.topk(m, dim=1)
        confusion_topk = [(indices[values > 0]).tolist() for values, indices in zip(top_m_values, top_m_indices)]
        metrics['confusion_topk'] = confusion_topk

    return metrics

def build_class_enemble(class_names, baseline='', before_text='', between_text=', ', after_text='', apply_descriptor_modification=False):
    class_names_enemble = [None] * len(class_names)
    if baseline == '':
        for i in range(len(class_names)):
            if ':' in class_names[i]:
                word_lists = class_names[i].split(':')[1].strip().split(', ')
                word_to_add = class_names[i].split(':')[0].strip().replace('_', ' ')
                build_descriptor_string = lambda item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, apply_descriptor_modification)}{after_text}"
                class_names_enemble[i] = [build_descriptor_string(word.replace('_', ' ')) for word in word_lists]
            else:
                class_names_enemble[i] = [before_text + class_names[i].replace('_', ' ')]
    elif baseline == 'CuPL':
        for i, (k, v) in enumerate(class_names.items()):
            if len(v) == 0:
                class_names_enemble[i] = [k]
            else:
                class_names_enemble[i] = [word for word in v]
    elif baseline == 'DCLIP':
        for i, (k, v) in enumerate(class_names.items()):
            if len(v) == 0:
                class_names_enemble[i] = [before_text + wordify(k) + after_text]
            else:
                class_names_enemble[i] = [before_text + wordify(k) + between_text + modify_descriptor(word, apply_descriptor_modification) + after_text for word in v]
    elif baseline == 'WaffleCLIP':
        import pickle as pkl
        waffle_count = 15
        word_list = pkl.load(open('baseline/WaffleCLIP/word_list.pkl', 'rb'))
        
        key_list = list(class_names.keys())
        descr_list = [list(values) for values in class_names.values()]
        descr_list = np.array([x for y in descr_list for x in y])
        structured_descriptor_builder = lambda item, cls: f"{before_text}{cls}{between_text}{modify_descriptor(item, apply_descriptor_modification)}{after_text}"

        avg_num_words = int(np.max([np.round(np.mean([len(wordify(x).split(' ')) for x in key_list])), 1]))
        avg_word_length = int(np.round(np.mean([np.mean([len(y) for y in wordify(x).split(' ')]) for x in key_list])))        
        word_list = [x[:avg_word_length] for x in word_list]

        # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
        character_list = [x.split(' ') for x in descr_list]
        character_list = [x.replace(',', '').replace('.', '') for x in np.unique([x for y in character_list for x in y])]
        character_list = np.unique(list(''.join(character_list)))
        
        num_spaces = int(np.round(np.mean([np.sum(np.array(list(x)) == ' ') for x in key_list]))) + 1
        num_chars = int(np.ceil(np.mean([np.max([len(y) for y in x.split(' ')]) for x in key_list])))            
        num_chars += num_spaces - num_chars%num_spaces
        
        sample_key = ''
        for s in range(num_spaces):
            for _ in range(num_chars//num_spaces):
                sample_key += 'a'
            if s < num_spaces - 1:
                sample_key += ' '
                
        base_gpt_descriptions = {key: items for key, items in class_names.items()}
        all_descr = [values for values in base_gpt_descriptions.values()]
        all_descr = [x for y in all_descr for x in y]
        gpt_descriptions = {key: [] for key in class_names.keys()}

        effective_waffle_count = int(2/3 * waffle_count)        
        
        for key in key_list:
            for sc in range(effective_waffle_count):
                base_word = ''                
                for a in range(avg_num_words):
                    base_word += np.random.choice(word_list, 1, replace=False)[0]
                    if a < avg_num_words - 1:
                        base_word += ' '
                gpt_descriptions[key].append(structured_descriptor_builder(base_word, key))
                noise_word = ''                
                for c in sample_key:
                    if c != ' ':
                        noise_word += np.random.choice(character_list, 1, replace=False)[0]
                    else:
                        noise_word += ', '
                gpt_descriptions[key].append(structured_descriptor_builder(noise_word, key))

        match_key = np.random.choice(key_list)
        gpt_descriptions = {key: gpt_descriptions[match_key] for key in key_list}
        for key in gpt_descriptions:
            gpt_descriptions[key] = [x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]]
        
        # For every random character and word descriptor pair, we add a GPT descriptor
        # sampled from the list of available descriptors.
        for key in key_list:
            if len(base_gpt_descriptions[key]) == 0:
                continue
            for sc in range(effective_waffle_count):
                word = np.random.choice(base_gpt_descriptions[key], 1)[0]
                gpt_descriptions[key].append(structured_descriptor_builder(word, key))
                word = np.random.choice(base_gpt_descriptions[key], 1)[0]
                gpt_descriptions[key].append(structured_descriptor_builder(word, key))
        
        # To ensure the correct number of random word sequences, random character sequences and GPT descriptions, we
        # subsample for each class individually. This does result in slightly different descriptor distributions per class.
        for key in key_list:
            if len(gpt_descriptions[key]) > effective_waffle_count * 3:
                gpt_descriptions[key] = list(np.random.choice(gpt_descriptions[key], effective_waffle_count * 3, replace=False))
            else:
                gpt_descriptions[key] = list(np.random.choice(gpt_descriptions[key], effective_waffle_count * 3, replace=True))
        class_names_enemble = [gpt_descriptions[key] for key in key_list]
    
    return class_names_enemble

def build_visual_feedback(visual_type, metric, class_names):
    if visual_type == 'confusion_topk' or visual_type == 'confusion_thresh':
        visual_feedback = ""
        for i in range(len(metric[visual_type])):
            if len(metric[visual_type][i]) == 0:
                visual_feedback += f"{class_names[i]} is not confused with any other classes\n"
            else:
                visual_feedback += f"{class_names[i]} is confused with: {', '.join([class_names[j] for j in metric[visual_type][i]])}\n" 
    elif visual_type == 'confusion_matrix':
        visual_feedback = str(metric[visual_type])
    else:
        visual_feedback = ""
        
    return visual_feedback

def build_memory_bank(bank, class_names, cluster_idx):
    memory_bank = ""
    for i in cluster_idx:
        if len(bank[i]) == 0:
            memory_bank += f"{class_names[i]}: None\n"
        else:
            memory_bank += f"{class_names[i]}: {', '.join(bank[i])}\n"
    return memory_bank

def list2str(lst):
    return "[\n" + ",\n".join(['    "{}"'.format(item) for item in lst]) + "\n]"