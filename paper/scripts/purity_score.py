"""Purity score for Memory Wrap checkpoints.

Two metrics per query, both in [0, 1]:
  soft      = sum of sparsemax attention weights on same-class memory items.
              Measures how much the model attends to correct-class examples.
  coherence = fraction of sparsemax-selected items belonging to the dominant
              class in the support set, regardless of whether that class matches
              the query label. Accuracy-independent: a support set that is all
              one class (even if wrong) scores 1.0. Analogous to PIPNet's
              "purest part" scoring which takes the max over parts rather than
              requiring the part to match the query annotation.

Pass --compare_path to a second checkpoint to compute coherence restricted to
queries where BOTH models predict correctly. This controls for example
difficulty and isolates retrieval quality.

Averaged over `--num_redraws` random memory draws per checkpoint, then
mean/std across checkpoints in the directory. Mirrors paper/eval.py.
"""
import os
import sys
import warnings
# Make `utils` importable regardless of cwd by anchoring on this file's dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Silence noisy torch FutureWarnings (weights_only default, deprecated AMP APIs, etc.).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import time

import absl.app
import absl.flags
import numpy as np
import torch

import utils.datasets as datasets
import utils.utils as utils
import utils.tracking as tracking

absl.flags.DEFINE_string("path", None, "Dir of .pt files or single .pt path.")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "Datasets directory.")
absl.flags.DEFINE_integer("num_redraws", 5, "Random memory draws per checkpoint.")
absl.flags.DEFINE_integer("max_images", 0,
    "Max test images to evaluate. 0 = all. When > 0 the test loader is "
    "rebuilt with shuffle=True and --seed_test so the same subset is used "
    "across runs (matches generate_memory_images.py).")
absl.flags.DEFINE_integer("seed_test", 42,
    "Seed for the shuffled test-loader when --max_images > 0.")
absl.flags.DEFINE_boolean("diagnose", False,
    "Print support-set sparsity and purity broken down by correct/wrong predictions.")
absl.flags.DEFINE_string("compare_path", None,
    "Path to a second checkpoint (or dir). When set, coherence is computed only "
    "on queries where BOTH this model and --path model predict correctly, "
    "controlling for example difficulty.")
absl.flags.mark_flag_as_required("path")
FLAGS = absl.flags.FLAGS


def _coherence_per_query(w, m_lbl, num_classes, device):
    """Return per-query coherence tensor for queries with non-zero support."""
    nonzero = w > 0  # (B, M)
    has_support = nonzero.any(dim=1)  # (B,)
    if not has_support.any():
        return None, has_support
    support_size = nonzero.float().sum(dim=1)  # (B,)
    m_lbl_oh = torch.zeros(m_lbl.size(0), num_classes, device=device).scatter_(
        1, m_lbl.unsqueeze(1), 1.0)  # (M, C)
    class_counts = torch.mm(nonzero.float(), m_lbl_oh)  # (B, C)
    dominant = class_counts.max(dim=1).values  # (B,)
    per_query = dominant / support_size.clamp(min=1)  # (B,)
    return per_query, has_support


def _focus_per_query(w, q_feat, m_feat):
    """Mean cosine similarity between query and each support-set item.

    Unlike coherence, this is label-free: it measures how visually tight
    the retrieval is — are the retrieved items actually close to the query
    in feature space, or just the same class but visually spread out?
    SupCon's contrastive training pulls similar-looking items together, so
    its support items should sit much closer to the query.
    """
    nonzero = w > 0  # (B, M)
    has_support = nonzero.any(dim=1)
    if not has_support.any():
        return None, has_support
    # normalise to unit vectors for cosine similarity
    q_norm = q_feat / (q_feat.norm(dim=1, keepdim=True) + 1e-8)  # (B, D)
    m_norm = m_feat / (m_feat.norm(dim=1, keepdim=True) + 1e-8)  # (M, D)
    cos_sim = torch.mm(q_norm, m_norm.t())  # (B, M)  values in [-1, 1]
    support_size = nonzero.float().sum(dim=1).clamp(min=1)  # (B,)
    # sum cosine sims over selected items, divide by support size
    per_query = (cos_sim * nonzero.float()).sum(dim=1) / support_size  # (B,)
    return per_query, has_support


def diagnose(model, test_loader, mem_loader, device, max_images=0):
    """Print sparsity and retrieved purity split by correct vs. wrong predictions."""
    model.eval()
    support_sizes = []
    purity_correct, purity_wrong = [], []
    with torch.no_grad():
        mem_iter = iter(mem_loader)
        n = 0
        for q_imgs, q_lbl in test_loader:
            if max_images > 0 and n >= max_images:
                break
            if max_images > 0 and n + q_imgs.size(0) > max_images:
                keep = max_images - n
                q_imgs, q_lbl = q_imgs[:keep], q_lbl[:keep]
            try:
                m_imgs, m_lbl = next(mem_iter)
            except StopIteration:
                mem_iter = iter(mem_loader); m_imgs, m_lbl = next(mem_iter)
            q_imgs, q_lbl = q_imgs.to(device), q_lbl.to(device)
            m_imgs, m_lbl = m_imgs.to(device), m_lbl.to(device)

            q_feat = model.forward_encoder(q_imgs)
            m_feat = model.forward_encoder(m_imgs)
            match = (q_lbl.unsqueeze(1) == m_lbl.unsqueeze(0)).float()

            logits, w = model.mw(q_feat, m_feat, return_weights=True)
            pred = logits.argmax(dim=1)
            correct_mask = (pred == q_lbl)

            nonzero = w > 0
            support_size = nonzero.float().sum(dim=1)
            support_sizes.extend(support_size.cpu().tolist())

            has_support = nonzero.any(dim=1)
            if has_support.any():
                same_class = (match * nonzero.float()).sum(dim=1) / support_size.clamp(min=1)
                for i in range(q_imgs.size(0)):
                    if not nonzero[i].any():
                        continue
                    if correct_mask[i]:
                        purity_correct.append(same_class[i].item())
                    else:
                        purity_wrong.append(same_class[i].item())
            n += q_imgs.size(0)

    print(f"\n--- DIAGNOSIS (n={n} queries, one memory draw) ---")
    print(f"Mean support-set size : {np.mean(support_sizes):.2f} +/- {np.std(support_sizes):.2f}  "
          f"(min={np.min(support_sizes):.0f}, max={np.max(support_sizes):.0f})")
    if purity_correct:
        print(f"Retrieved purity | correct preds (n={len(purity_correct)}): "
              f"{np.mean(purity_correct):.4f} +/- {np.std(purity_correct):.4f}")
    if purity_wrong:
        print(f"Retrieved purity | wrong   preds (n={len(purity_wrong)}): "
              f"{np.mean(purity_wrong):.4f} +/- {np.std(purity_wrong):.4f}")
    total = len(purity_correct) + len(purity_wrong)
    if total:
        print(f"Retrieved purity | all     preds (n={total}): "
              f"{np.mean(purity_correct + purity_wrong):.4f} +/- "
              f"{np.std(purity_correct + purity_wrong):.4f}")
    print("---\n")


def purity(model, test_loader, mem_loader, num_redraws, device, max_images=0):
    """Compute Soft and Coherence averaged over all queries."""
    model.eval()
    soft, coherence = [], []
    num_classes = None
    with torch.no_grad():
        for _ in range(num_redraws):
            mem_iter = iter(mem_loader)
            s_sum, c_sum, c_n, n = 0.0, 0.0, 0, 0
            for q_imgs, q_lbl in test_loader:
                if max_images > 0 and n >= max_images:
                    break
                if max_images > 0 and n + q_imgs.size(0) > max_images:
                    keep = max_images - n
                    q_imgs = q_imgs[:keep]
                    q_lbl = q_lbl[:keep]
                try:
                    m_imgs, m_lbl = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(mem_loader); m_imgs, m_lbl = next(mem_iter)
                q_imgs, q_lbl = q_imgs.to(device), q_lbl.to(device)
                m_imgs, m_lbl = m_imgs.to(device), m_lbl.to(device)

                if num_classes is None:
                    num_classes = int(m_lbl.max().item()) + 1

                q_feat = model.forward_encoder(q_imgs)
                m_feat = model.forward_encoder(m_imgs)
                match = (q_lbl.unsqueeze(1) == m_lbl.unsqueeze(0)).float()

                _, w = model.mw(q_feat, m_feat, return_weights=True)
                s_sum += (w * match).sum().item()

                per_query_coh, has_support = _coherence_per_query(w, m_lbl, num_classes, device)
                if per_query_coh is not None:
                    c_sum += per_query_coh[has_support].sum().item()
                    c_n += has_support.sum().item()

                n += q_imgs.size(0)

            soft.append(s_sum / n)
            coherence.append(c_sum / c_n if c_n > 0 else float('nan'))
    return soft, coherence


def _soft_per_query(w, q_lbl, m_lbl):
    """Soft purity per query: attention mass on same-class memory items."""
    match = (q_lbl.unsqueeze(1) == m_lbl.unsqueeze(0)).float()  # (B, M)
    return (w * match).sum(dim=1)  # (B,)


def compare_coherence(model_a, model_b, test_loader, mem_loader,
                      num_redraws, device, max_images=0):
    """Coherence, focus, and soft purity for each model restricted to queries
    both models get correct.

    Both models see the same raw test images and the same raw memory images.
    Each model encodes them independently with its own encoder. Only queries
    where both predict correctly contribute to the score, controlling for
    example difficulty.
    """
    model_a.eval()
    model_b.eval()
    coh_a, coh_b, foc_a, foc_b, sft_a, sft_b, n_both_correct = [], [], [], [], [], [], []
    num_classes = None
    with torch.no_grad():
        for _ in range(num_redraws):
            mem_iter = iter(mem_loader)
            ca_sum, cb_sum, fa_sum, fb_sum, sa_sum, sb_sum, cn, n = \
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
            for q_imgs, q_lbl in test_loader:
                if max_images > 0 and n >= max_images:
                    break
                if max_images > 0 and n + q_imgs.size(0) > max_images:
                    keep = max_images - n
                    q_imgs = q_imgs[:keep]
                    q_lbl = q_lbl[:keep]
                try:
                    m_imgs, m_lbl = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(mem_loader); m_imgs, m_lbl = next(mem_iter)

                q_imgs, q_lbl = q_imgs.to(device), q_lbl.to(device)
                m_imgs, m_lbl = m_imgs.to(device), m_lbl.to(device)

                if num_classes is None:
                    num_classes = int(m_lbl.max().item()) + 1

                # each model encodes the same raw images with its own encoder
                q_feat_a = model_a.forward_encoder(q_imgs)
                m_feat_a = model_a.forward_encoder(m_imgs)
                logits_a, w_a = model_a.mw(q_feat_a, m_feat_a, return_weights=True)

                q_feat_b = model_b.forward_encoder(q_imgs)
                m_feat_b = model_b.forward_encoder(m_imgs)
                logits_b, w_b = model_b.mw(q_feat_b, m_feat_b, return_weights=True)

                correct_a = logits_a.argmax(dim=1) == q_lbl
                correct_b = logits_b.argmax(dim=1) == q_lbl
                both_correct = correct_a & correct_b

                if both_correct.any():
                    per_coh_a, has_a = _coherence_per_query(w_a, m_lbl, num_classes, device)
                    per_coh_b, has_b = _coherence_per_query(w_b, m_lbl, num_classes, device)
                    per_foc_a, _ = _focus_per_query(w_a, q_feat_a, m_feat_a)
                    per_foc_b, _ = _focus_per_query(w_b, q_feat_b, m_feat_b)
                    per_sft_a = _soft_per_query(w_a, q_lbl, m_lbl)
                    per_sft_b = _soft_per_query(w_b, q_lbl, m_lbl)
                    mask = both_correct & has_a & has_b
                    if mask.any():
                        ca_sum += per_coh_a[mask].sum().item()
                        cb_sum += per_coh_b[mask].sum().item()
                        fa_sum += per_foc_a[mask].sum().item()
                        fb_sum += per_foc_b[mask].sum().item()
                        sa_sum += per_sft_a[mask].sum().item()
                        sb_sum += per_sft_b[mask].sum().item()
                        cn += mask.sum().item()

                n += q_imgs.size(0)

            coh_a.append(ca_sum / cn if cn > 0 else float('nan'))
            coh_b.append(cb_sum / cn if cn > 0 else float('nan'))
            foc_a.append(fa_sum / cn if cn > 0 else float('nan'))
            foc_b.append(fb_sum / cn if cn > 0 else float('nan'))
            sft_a.append(sa_sum / cn if cn > 0 else float('nan'))
            sft_b.append(sb_sum / cn if cn > 0 else float('nan'))
            n_both_correct.append(cn)
    return coh_a, coh_b, foc_a, foc_b, sft_a, sft_b, n_both_correct


def _load_model_and_loaders(path, dataset_dir, device):
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path) if f.endswith('.pt'))
        base = path
    else:
        names = [os.path.basename(path)]
        base = os.path.dirname(path)
    entries = []
    for name in names:
        run = int(name.split('.')[0]) - 1
        utils.set_seed(run)
        ckpt = torch.load(os.path.join(base, name), map_location=device)
        model = utils.get_model(ckpt['model_name'], ckpt['num_classes'],
                                model_type=ckpt['modality'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        _, _, test_loader, mem_loader = getattr(datasets, 'get_' + ckpt['dataset_name'])(
            dataset_dir, batch_size_train=128, batch_size_test=500,
            batch_size_memory=ckpt['mem_examples'],
            size_train=ckpt['train_examples'], seed=run,
        )
        entries.append((run, model, test_loader, mem_loader, ckpt))
    return entries


def run_experiment(path, dataset_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    entries_a = _load_model_and_loaders(path, dataset_dir, device)
    tracker = tracking.init(
        name=os.path.normpath(path), group='purity', job_type='purity',
        config={'path': path, 'compare_path': FLAGS.compare_path,
                'num_redraws': FLAGS.num_redraws, 'max_images': FLAGS.max_images})

    # --- comparison mode ---
    if FLAGS.compare_path:
        entries_b = _load_model_and_loaders(FLAGS.compare_path, dataset_dir, device)
        if len(entries_a) != len(entries_b):
            print(f"Warning: --path has {len(entries_a)} checkpoints, "
                  f"--compare_path has {len(entries_b)}. Using min.", flush=True)
        pairs = list(zip(entries_a, entries_b))

        all_coh_a, all_coh_b, all_foc_a, all_foc_b, all_sft_a, all_sft_b = \
            [], [], [], [], [], []
        for (run_a, model_a, test_loader_a, mem_loader_a, ckpt_a), \
            (run_b, model_b, test_loader_b, mem_loader_b, ckpt_b) in pairs:

            if FLAGS.max_images > 0:
                gen = torch.Generator().manual_seed(FLAGS.seed_test)
                shared_test_loader = torch.utils.data.DataLoader(
                    test_loader_a.dataset, batch_size=500, shuffle=True, generator=gen)
            else:
                shared_test_loader = test_loader_a

            t0 = time.time()
            # use model_a's mem_loader as the shared memory source — both models
            # encode the same raw images with their own encoders
            coh_a, coh_b, foc_a, foc_b, sft_a, sft_b, ns = compare_coherence(
                model_a, model_b, shared_test_loader,
                mem_loader_a,
                FLAGS.num_redraws, device, max_images=FLAGS.max_images)
            mean_ca, mean_cb = float(np.nanmean(coh_a)), float(np.nanmean(coh_b))
            mean_fa, mean_fb = float(np.nanmean(foc_a)), float(np.nanmean(foc_b))
            mean_sa, mean_sb = float(np.nanmean(sft_a)), float(np.nanmean(sft_b))
            all_coh_a.append(mean_ca)
            all_coh_b.append(mean_cb)
            all_foc_a.append(mean_fa)
            all_foc_b.append(mean_fb)
            all_sft_a.append(mean_sa)
            all_sft_b.append(mean_sb)
            print(f"Run:{run_a+1} | "
                  f"CohA:{mean_ca:.4f} CohB:{mean_cb:.4f} | "
                  f"FocA:{mean_fa:.4f} FocB:{mean_fb:.4f} | "
                  f"SoftA:{mean_sa:.4f} SoftB:{mean_sb:.4f} | "
                  f"BothCorrect(avg):{np.mean(ns):.0f}  E:{(time.time()-t0)/60:.2f}min",
                  flush=True)
            tracker.log({'seed': run_a, 'coherence/path': mean_ca, 'coherence/compare': mean_cb,
                         'focus/path': mean_fa, 'focus/compare': mean_fb,
                         'soft/path': mean_sa, 'soft/compare': mean_sb})

        print(f"\nSUMMARY (n={len(all_coh_a)}, both-correct queries only) | "
              f"Coherence --path: {np.nanmean(all_coh_a):.4f} +/- {np.nanstd(all_coh_a):.4f} | "
              f"Coherence --compare_path: {np.nanmean(all_coh_b):.4f} +/- {np.nanstd(all_coh_b):.4f} | "
              f"Focus --path: {np.nanmean(all_foc_a):.4f} +/- {np.nanstd(all_foc_a):.4f} | "
              f"Focus --compare_path: {np.nanmean(all_foc_b):.4f} +/- {np.nanstd(all_foc_b):.4f} | "
              f"Soft --path: {np.nanmean(all_sft_a):.4f} +/- {np.nanstd(all_sft_a):.4f} | "
              f"Soft --compare_path: {np.nanmean(all_sft_b):.4f} +/- {np.nanstd(all_sft_b):.4f}",
              flush=True)
        for metric, values in [('coherence/path', all_coh_a), ('coherence/compare', all_coh_b),
                               ('focus/path', all_foc_a), ('focus/compare', all_foc_b),
                               ('soft/path', all_sft_a), ('soft/compare', all_sft_b)]:
            tracker.summary[f'{metric}_mean'] = float(np.nanmean(values))
            tracker.summary[f'{metric}_std'] = float(np.nanstd(values))
        tracker.finish()
        return

    # --- single-model mode ---
    if FLAGS.max_images > 0:
        gen = torch.Generator().manual_seed(FLAGS.seed_test)

    run_soft, run_coherence = [], []
    for run, model, test_loader, mem_loader, ckpt in entries_a:
        if FLAGS.max_images > 0:
            gen = torch.Generator().manual_seed(FLAGS.seed_test)
            test_loader = torch.utils.data.DataLoader(
                test_loader.dataset, batch_size=500, shuffle=True, generator=gen)

        if FLAGS.diagnose:
            diagnose(model, test_loader, mem_loader, device, max_images=FLAGS.max_images)

        t0 = time.time()
        soft, coherence = purity(model, test_loader, mem_loader,
                                 FLAGS.num_redraws, device, max_images=FLAGS.max_images)
        run_soft.append(float(np.mean(soft)))
        run_coherence.append(float(np.nanmean(coherence)))

        print(f"Run:{run+1} | Soft:{run_soft[-1]:.4f} | Coherence:{run_coherence[-1]:.4f}"
              f"  E:{(time.time()-t0)/60:.2f}min", flush=True)
        tracker.log({'seed': run, 'soft': run_soft[-1], 'coherence': run_coherence[-1]})

    print(f"SUMMARY (n={len(run_soft)}) | "
          f"Soft: {np.mean(run_soft):.4f} +/- {np.std(run_soft):.4f} | "
          f"Coherence: {np.nanmean(run_coherence):.4f} +/- {np.nanstd(run_coherence):.4f}",
          flush=True)
    tracker.summary['soft_mean'] = float(np.mean(run_soft))
    tracker.summary['soft_std'] = float(np.std(run_soft))
    tracker.summary['coherence_mean'] = float(np.nanmean(run_coherence))
    tracker.summary['coherence_std'] = float(np.nanstd(run_coherence))
    tracker.finish()


def main(argv=None):
    run_experiment(FLAGS.path, FLAGS.dir_dataset)


if __name__ == '__main__':
    absl.app.run(main)
