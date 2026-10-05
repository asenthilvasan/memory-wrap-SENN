"""Memory sets of several Memory Wrap checkpoints on the same queries.

Every checkpoint sees the same test queries and the same memory set (drawn
from the training subset of the checkpoints' shared seed), so differences
come from the models alone.

--layout=grid (default): one figure per checkpoint in the Memory Wrap paper's
style. Each column is a random test query with the model's prediction above
and, below, every memory with positive sparsemax weight ("Used Samples"),
highest weight first. Saved as <out stem>_<name>.png. A model without a
memory (modality std, e.g. Scratch + Linear) gets its --neighbours nearest
memory images by cosine similarity of its penultimate features instead,
titled "Nearest neighbours": they show what its feature space considers
similar, but the model does not use them to predict.

--layout=strip: one figure with a row per checkpoint and query showing the
top_k memories, green border = same class as the query, red = other class,
plus per-query soft purity and coherence (as in purity_score.py). Queries are
chosen among those every model classifies correctly: one at random and one
with the largest spread in soft purity across models (selected for contrast).

Usage (from paper/):
    python scripts/compare_memory_sets.py --dir_dataset=datasets \\
        --paths=A/1.pt,B/1.pt,C/1.pt --names="Scratch + MW,SupCon MW frozen,SupCon MW fine-tuned" \\
        --out=../plan/svhn_eval_memory_sets.png
"""
import os
import sys
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import absl.app
import absl.flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import utils.datasets as datasets
import utils.utils as utils

absl.flags.DEFINE_string("paths", None, "Comma-separated checkpoint files (same seed).")
absl.flags.DEFINE_string("names", None, "Comma-separated display names, one per checkpoint.")
absl.flags.DEFINE_string("dir_dataset", 'datasets', "Datasets directory.")
absl.flags.DEFINE_string("out", 'memory_sets.png', "Output image path.")
absl.flags.DEFINE_enum("layout", "grid", ["grid", "strip"], "Figure layout (see module docstring).")
absl.flags.DEFINE_integer("num_queries", 3, "Random queries per figure (grid layout).")
absl.flags.DEFINE_integer("neighbours", 12, "Nearest memories shown for models without a memory (grid layout).")
absl.flags.DEFINE_integer("top_k", 10, "Memories shown per model and query (strip layout).")
absl.flags.DEFINE_integer("mem_seed", 0, "Seed for the shared memory draw.")
absl.flags.DEFINE_integer("query_seed", 0, "Seed for choosing the random query.")
absl.flags.DEFINE_integer("max_queries", 2000, "Test images scanned for candidate queries.")
absl.flags.mark_flag_as_required("paths")
FLAGS = absl.flags.FLAGS


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    if ckpt.get('val_examples', 0):
        raise ValueError(f'{path} is a validation-search checkpoint.')
    model = utils.get_model(ckpt['model_name'], ckpt['num_classes'], model_type=ckpt['modality'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt


def features(model, ckpt, x):
    """(logits, penultimate features) for Memory Wrap and plain models."""
    if ckpt['modality'] == 'std':
        captured = {}
        hook = model.linear.register_forward_hook(lambda m, inp, out: captured.update(f=inp[0]))
        logits = model(x)
        hook.remove()
        return (logits[0] if isinstance(logits, tuple) else logits), captured['f']
    return None, model.forward_encoder(x)


def nearest_as_weights(q_feat, m_feat, k):
    """Rank-based pseudo-weights that select the k nearest memories (cosine),
    so plain models plug into the same plotting code."""
    sim = torch.nn.functional.normalize(q_feat, dim=1) @ torch.nn.functional.normalize(m_feat, dim=1).t()
    w = torch.zeros_like(sim)
    top = sim.topk(k, dim=1).indices
    w.scatter_(1, top, torch.arange(k, 0, -1, device=sim.device, dtype=w.dtype).expand(len(sim), k))
    return w


def grid_figures(names, weights, preds, q_imgs, q_lbls, m_imgs, undo, plain):
    """One Memory Wrap paper style figure per model on the same random queries."""
    rng = np.random.default_rng(FLAGS.query_seed)
    queries = sorted(rng.choice(len(q_lbls), FLAGS.num_queries, replace=False).tolist())
    # Same grid size everywhere so tiles render at the same scale; empty slots
    # stay black. Four columns as in the paper, wider if a support set is large.
    most = max(int((w[q] > 0).sum()) for w in weights for q in queries)
    nrow = max(4, int(np.ceil(np.sqrt(most))))
    slots = nrow * int(np.ceil(most / nrow))
    tiles = undo(m_imgs.cpu()).clamp(0, 1)
    stem, ext = os.path.splitext(FLAGS.out)
    for k, name in enumerate(names):
        fig, axes = plt.subplots(2, len(queries), figsize=(2.8 * len(queries), 6.2), squeeze=False,
                                 gridspec_kw={'hspace': 0.2, 'wspace': 0.15})
        for j, q in enumerate(queries):
            ax = axes[0, j]
            ax.imshow(undo(q_imgs[q].cpu()).clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_title(f'Prediction:{int(preds[k][q])}  (label {int(q_lbls[q])})', fontsize=10)
            ax.axis('off')
            w = weights[k][q].cpu()
            used = torch.argsort(w, descending=True)[:int((w > 0).sum())]
            grid_in = torch.zeros(slots, *tiles.shape[1:])
            grid_in[:len(used)] = tiles[used]
            grid = torchvision.utils.make_grid(grid_in, nrow=nrow, padding=2, pad_value=0)
            ax = axes[1, j]
            ax.imshow(grid.permute(1, 2, 0).numpy())
            ax.set_title(f'Nearest neighbours ({len(used)})' if plain[k] else f'Used Samples ({len(used)})',
                         fontsize=10)
            ax.axis('off')
        fig.suptitle(name, x=0.02, ha='left', fontsize=14, fontstyle='italic')
        slug = ''.join(c if c.isalnum() else '_' for c in name.lower()).strip('_')
        path = f'{stem}_{slug}{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')
    print(f'Queries {queries}: labels {[int(q_lbls[q]) for q in queries]}; ' + '; '.join(
        f'{n}: preds {[int(preds[k][q]) for q in queries]}, support {[int((weights[k][q] > 0).sum()) for q in queries]}'
        for k, n in enumerate(names)))


def main(argv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = FLAGS.paths.split(',')
    names = FLAGS.names.split(',') if FLAGS.names else paths
    loaded = [load(p, device) for p in paths]
    ckpt = loaded[0][1]
    for _, c in loaded:
        if (c['seed'], c['train_examples'], c['dataset_name']) != (ckpt['seed'], ckpt['train_examples'], ckpt['dataset_name']):
            raise ValueError('Checkpoints must share seed, train_examples and dataset.')

    # Same subset as training for this seed; one fixed memory draw for all models.
    _, _, test_loader, mem_loader = getattr(datasets, 'get_' + ckpt['dataset_name'])(
        FLAGS.dir_dataset, batch_size_train=64, batch_size_test=500,
        batch_size_memory=ckpt['mem_examples'], size_train=ckpt['train_examples'], seed=ckpt['seed'])
    train_subset = mem_loader.dataset
    gen = torch.Generator().manual_seed(FLAGS.mem_seed)
    mem_idx = torch.randperm(len(train_subset), generator=gen)[:ckpt['mem_examples']]
    mem = [train_subset[int(i)] for i in mem_idx]
    m_imgs = torch.stack([m[0] for m in mem]).to(device)
    m_lbl = torch.tensor([m[1] for m in mem], device=device)

    # Weights and predictions of every model for the first max_queries test images.
    plain = [c['modality'] == 'std' for _, c in loaded]
    if any(plain) and FLAGS.layout == 'strip':
        raise ValueError('The strip layout needs Memory Wrap checkpoints only.')
    weights, preds, q_imgs, q_lbls = [[] for _ in loaded], [[] for _ in loaded], [], []
    seen = 0
    with torch.no_grad():
        m_feats = [features(model, c, m_imgs)[1] for model, c in loaded]
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            for k, (model, c) in enumerate(loaded):
                logits, q_feat = features(model, c, x)
                if plain[k]:
                    w = nearest_as_weights(q_feat, m_feats[k], FLAGS.neighbours)
                else:
                    logits, w = model.mw(q_feat, m_feats[k], return_weights=True)
                weights[k].append(w.float())
                preds[k].append(logits.argmax(dim=1))
            q_imgs.append(x.cpu())
            q_lbls.append(y)
            seen += x.size(0)
            if seen >= FLAGS.max_queries:
                break
    weights = [torch.cat(w) for w in weights]
    preds = [torch.cat(p) for p in preds]
    q_imgs, q_lbls = torch.cat(q_imgs), torch.cat(q_lbls)

    undo = getattr(datasets, 'undo_normalization_' + ckpt['dataset_name'])
    if FLAGS.layout == 'grid':
        grid_figures(names, weights, preds, q_imgs, q_lbls, m_imgs, undo, plain)
        return

    match = (q_lbls.unsqueeze(1) == m_lbl.unsqueeze(0)).float()
    soft = torch.stack([(w * match).sum(dim=1) for w in weights])  # (models, Q)
    onehot = torch.nn.functional.one_hot(m_lbl, ckpt['num_classes']).float()
    coherence = torch.stack([(torch.mm((w > 0).float(), onehot).max(dim=1).values /
                              (w > 0).float().sum(dim=1).clamp(min=1)) for w in weights])
    all_correct = torch.stack([p == q_lbls for p in preds]).all(dim=0)
    candidates = all_correct.nonzero().squeeze(1).cpu()
    if len(candidates) == 0:
        raise RuntimeError('No query that every model classifies correctly.')
    rng = np.random.default_rng(FLAGS.query_seed)
    random_q = int(candidates[rng.integers(len(candidates))])
    spread = (soft.max(dim=0).values - soft.min(dim=0).values).cpu()
    contrast_q = int(candidates[spread[candidates].argmax()])
    queries = [(random_q, 'random query (all models correct)'),
               (contrast_q, 'largest soft-purity spread (selected for contrast)')]

    to_img = lambda t: undo(t.cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    k_show = FLAGS.top_k
    # One block of rows per query, separated by a thin spacer row for its caption.
    block = len(loaded) + 1
    rows = len(queries) * block
    ratios = ([0.35] + [1] * len(loaded)) * len(queries)
    fig, axes = plt.subplots(rows, k_show + 1, figsize=(1.0 * (k_show + 1) + 2.6, 1.15 * sum(ratios) + 0.4),
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.5, 'height_ratios': ratios})
    for qi, (q, caption) in enumerate(queries):
        for ax in axes[qi * block]:
            ax.axis('off')
        axes[qi * block, 1].text(0, 0.2, caption, transform=axes[qi * block, 1].transAxes,
                                 fontsize=8, fontweight='bold')
        for mi, name in enumerate(names):
            r = qi * block + 1 + mi
            ax = axes[r, 0]
            ax.imshow(to_img(q_imgs[q]))
            ax.set_xticks([]); ax.set_yticks([])
            if mi == 0:
                ax.set_title(f'query, label {int(q_lbls[q])}', fontsize=7)
            w = weights[mi][q]
            order = torch.argsort(w, descending=True)
            support = int((w > 0).sum())
            more = f', +{support - k_show} more' if support > k_show else ''
            ax.set_ylabel(f'{name}\npred {int(preds[mi][q])} | support {support}{more}\n'
                          f'soft {soft[mi, q]:.2f} | coh {coherence[mi, q]:.2f}',
                          fontsize=6, rotation=0, ha='right', va='center', labelpad=4)
            for j in range(k_show):
                ax = axes[r, j + 1]
                ax.set_xticks([]); ax.set_yticks([])
                idx = int(order[j])
                if j >= support:
                    ax.axis('off')
                    continue
                ax.imshow(to_img(m_imgs[idx]))
                same = int(m_lbl[idx]) == int(q_lbls[q])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1a9850' if same else '#d73027')
                    spine.set_linewidth(2.2)
                ax.set_title(f'{float(w[idx]):.2f} · {int(m_lbl[idx])}', fontsize=6, pad=2)
    fig.suptitle(f'Top-{k_show} memories by attention weight (title: weight · memory label). '
                 f'Same query and same {ckpt["mem_examples"]}-image memory set for every model, seed {ckpt["seed"]}. '
                 'Green = query class, red = other class.', fontsize=7, y=0.995)
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    fig.savefig(FLAGS.out, dpi=200, bbox_inches='tight')
    print(f'Saved {FLAGS.out}; queries {random_q} (random), {contrast_q} (contrast); '
          f'{len(candidates)} of {len(q_lbls)} scanned queries are correct for every model.')
    for q, caption in queries:
        print(caption, ' | '.join(f'{n}: soft {soft[k, q]:.2f} coh {coherence[k, q]:.2f} '
                                  f'support {int((weights[k][q] > 0).sum())}' for k, n in enumerate(names)))


if __name__ == '__main__':
    absl.app.run(main)
