"""Minimal contrastive pretraining for Memory Wrap encoders.

Supports three objectives:
  - supcon  (Khosla et al., 2020): class-label-supervised. All same-class
    features are positives. Biases retrieval toward same-class memories.
  - simclr  (Chen et al., 2020): self-supervised, no labels. Only positive
    for each anchor is the OTHER augmented view of the same image. Biases
    retrieval toward visually similar memories regardless of class.
  - hybrid: weighted sum  alpha * supcon + (1-alpha) * simclr.  Produces a
    hierarchical feature geometry: tightest clusters around each individual
    image (augmentation-invariant), medium clusters around each class,
    far separation between classes. "Looks similar AND same-class."

All three train `forward_encoder` so that cos(f(x), f(y)) is larger for
pairs the loss considers positive. The downstream effect is in Memory
Wrap's `sparsemax(cos(encoder(query), encoder(memory_i)))` attention.

Usage:
    python pretrain_supcon.py                        # CIFAR10, supcon, mobilenet, FULL dataset
    python pretrain_supcon.py --dataset=SVHN         # SVHN instead of CIFAR10
    python pretrain_supcon.py --loss=simclr          # SimCLR (no labels)
    python pretrain_supcon.py --loss=hybrid          # SupCon + SimCLR (50/50)
    python pretrain_supcon.py --loss=hybrid --hybrid_alpha=0.7  # 70% supcon
    python pretrain_supcon.py --model=resnet18 --epochs=200
    # Same-data-budget pretraining: only see the 2000 images that downstream
    # run 3 (seed 3) will see. Pretrain one encoder per downstream seed.
    python pretrain_supcon.py --train_examples=2000 --seed=3
    # Enable the 2-layer MLP projection head (canonical SupCon/SimCLR).
    python pretrain_supcon.py --projection_dim=128

Output: models/<dataset>/{supcon,simclr,hybrid}/<model>/<train_examples or "full">/seed<seed>.pt
(config/pretrain_encoders.sh pretrains every seed of a sweep.)

Plug the encoders into downstream Memory Wrap training via:
    python train.py --modality=encoder_memory \\
        --pretrained_encoder=models/<dataset>/<loss>/<model>/<budget> \\
        --freeze_encoder=True
"""
import os
import sys
# absl is used for CLI flags to match the convention in train.py.
import absl.app, absl.flags
import torch
# Kubernetes pods default /dev/shm to 64MB, which is far too small for
# PyTorch DataLoader's default shared-memory tensor sharing. Switching to
# the 'file_system' strategy uses file descriptors instead and avoids the
# "unable to allocate shared memory" error on constrained pods.
torch.multiprocessing.set_sharing_strategy('file_system')
# F provides L2 normalization (F.normalize); we need unit vectors because
# SupCon works on cosine similarity = dot product of L2-normalized features.
import torch.nn.functional as F
from torchvision import datasets
# Reuse the existing model factory so SupCon-pretrained checkpoints use the
# exact same backbone as downstream Memory Wrap training.
import utils.utils as utils
import utils.tracking as tracking
# split_dataset implements the exact same seeded random_split used by the
# downstream Memory Wrap training pipeline (paper/utils/datasets.py). Reusing
# it here — with the same seed and per-dataset val_size — guarantees that
# when --train_examples matches config/train.yaml's train_examples, the
# pretraining sees the EXACT SAME image indices as downstream, i.e. a fair
# same-data-budget comparison instead of pretraining on the full dataset.
from utils.datasets import contrastive_augmentation, split_dataset


# --- CLI flags ---------------------------------------------------------------
# Default hyperparameters follow the SupCon paper's CIFAR-10 recipe.

# Which backbone to pretrain. Must be a key accepted by utils.get_model (e.g.
# 'mobilenet', 'resnet18', 'densenet', 'efficientnet', 'googlenet', ...).
absl.flags.DEFINE_string('model', 'mobilenet', 'Backbone (see utils/utils.py get_model)')
# 100 epochs is the typical CIFAR-10 SupCon budget; more helps slightly.
absl.flags.DEFINE_integer('epochs', 100, 'Pretraining epochs')
# SupCon benefits from large batches because more samples = more negatives
# per anchor = sharper contrastive signal. 256 is a single-GPU compromise.
absl.flags.DEFINE_integer('batch_size', 256, 'Pretraining batch size')
# Lower temperature = sharper softmax = harder negatives dominate the loss.
# 0.07 is the SupCon default; SimCLR's original paper used 0.5. Tune per task.
absl.flags.DEFINE_float('temperature', 0.07, 'Softmax temperature (0.07 supcon, 0.5 simclr typical)')
# Choice of contrastive objective. 'supcon' uses class labels as in Khosla et
# al. 2020; 'simclr' ignores labels and only treats the other view of the same
# image as a positive (Chen et al. 2020); 'hybrid' is a weighted sum of both.
# Retrieval behaviour: supcon -> same-class; simclr -> visually similar;
# hybrid -> both (tighter within-class, augmentation-invariant).
absl.flags.DEFINE_enum('loss', 'supcon', ['supcon', 'simclr', 'hybrid'],
                       'Contrastive objective.')
# Only used when --loss=hybrid. Weight on the SupCon term; (1-alpha) goes to
# the SimCLR term. 0.5 = equal. Higher -> more class-clustered; lower -> more
# visually-invariant.
absl.flags.DEFINE_float('hybrid_alpha', 0.5, 'SupCon weight in hybrid loss (0..1)')
# Large LR is standard for contrastive pretraining; cosine schedule below
# anneals it smoothly to zero. If you cut batch_size, cut lr proportionally.
absl.flags.DEFINE_float('lr', 0.5, 'Learning rate (SGD)')
# Linear LR warmup at the start of training. Matches the SupCon paper's
# reference implementation. CRITICAL when a projection head is used: a
# freshly-initialized 2-layer MLP + full lr=0.5 collapses features in the
# first few batches (loss gets stuck at log(2B-1) ~ 6.24 for batch=256),
# and once collapsed, gradients through the uniform softmax vanish and the
# run can't recover. Warmup gives the head a gentle start before the full
# LR kicks in. 10 epochs matches the SupCon CIFAR-10 recipe.
# Default 0 (disabled) matches the original working recipe. Warmup is mainly
# useful when --projection_dim > 0 (the freshly-initialized MLP head can't
# survive full LR on step 1), so enable it together with the projection
# head, not on its own.
absl.flags.DEFINE_integer('warmup_epochs', 0,
    'Linear LR warmup from ~0 up to --lr over this many epochs, then cosine '
    'decay. 0 (default) = plain cosine schedule from epoch 0. Set to ~10 '
    'when using --projection_dim > 0.')
absl.flags.DEFINE_string('data_dir', 'datasets', 'Dataset directory')
# Which image dataset to pretrain on. Both are 32x32 10-class datasets so the
# augmentation recipe and backbone architectures work for either, but they
# need different normalization stats and (for SVHN) no horizontal flip
# because flipped digits aren't digits.
absl.flags.DEFINE_enum('dataset', 'CIFAR10', ['CIFAR10', 'SVHN', 'CINIC10'],
                       'Dataset to pretrain on.')
# Data loading parallelism. With 2-view augmentation this pipeline is
# CPU-bound (each batch needs 2B independent RandomResizedCrop+ColorJitter
# passes). On a modern GPU (L4/L40/A100/H100) the default of 4 workers
# typically starves the GPU; 8-16 is a better starting point.
absl.flags.DEFINE_integer('num_workers', 8, 'DataLoader worker processes')
# Same-data-budget control. Default 0 means "use the full training set"
# (legacy behaviour). Set this to the same value as config/train.yaml's
# `train_examples` (e.g. 2000) to pretrain on exactly the same subset of
# images that downstream Memory Wrap training will see — this is the fair
# comparison when reporting "pretrained encoder + 2000-sample Memory Wrap"
# vs "scratch encoder + 2000-sample Memory Wrap": both stages then share
# the same data budget instead of letting pretraining cheat with 50k images.
absl.flags.DEFINE_integer('train_examples', 0,
    'Subset size to pretrain on (0 = full dataset). Match config/train.yaml '
    'train_examples for a same-data-budget comparison with downstream.')
# Seed controls which images end up in the subset and the encoder's initial
# weights. train.py runs seeds 0..runs-1 (saved as 1.pt..N.pt), each on its
# own subset, and loads the encoder whose seed matches. Using one encoder for
# every run leaks labels from its subset into the other runs.
absl.flags.DEFINE_integer('seed', 42,
    'Seed for the train/val split and weight init. Must equal the downstream '
    'run index (0..runs-1) whose subset this encoder is for.')
# Output dimension of the 2-layer MLP projection head applied on top of the
# encoder during pretraining. Standard SupCon/SimCLR practice (Khosla 2020,
# Chen 2020): apply the contrastive loss to the projection's output, NOT
# directly to the encoder features. The projection head absorbs the
# augmentation-invariance pressure of the contrastive loss so the encoder
# features `z = forward_encoder(x)` retain richer information for downstream
# tasks. The projection head is DISCARDED at checkpoint time --- only the
# encoder state_dict is saved, so downstream train.py is unchanged.
# Set to 0 to disable the projection head entirely (legacy behaviour, useful
# for ablation: "does the projection head matter for our setup?").
# Default is 0 (disabled) to reproduce the original working recipe. Enable
# this (e.g. --projection_dim=128) together with --projection_bn=True and
# an appropriate LR to experiment with the canonical SupCon/SimCLR setup.
# NOTE: in this repo, enabling the projection head at high LR (0.5) OR
# without BatchNorm has been observed to cause dimensional collapse (loss
# stuck at log(2B-1) ~ 6.24). Keep default off unless actively debugging.
absl.flags.DEFINE_integer('projection_dim', 0,
    'Output dim of the 2-layer MLP projection head used during pretraining. '
    '128 = SupCon/SimCLR canonical. 0 = disabled (default, apply loss '
    'directly to encoder features; matches the original working recipe).')
# Optional BatchNorm1d between the first Linear and the ReLU inside the
# projection head. NOT used by the SupCon/SimCLR reference recipes but IS
# used by BYOL, MoCo v3, DINO, and other methods. BN forces every hidden
# dimension to have mean 0 / unit std across the batch, which makes total
# dimensional collapse (all features landing on the same point/line of the
# 128-sphere) geometrically impossible. Enable this if you see loss stuck
# at log(2B-1) ~ 6.24 for batch=256 even after matching the SupCon LR.
absl.flags.DEFINE_bool('projection_bn', False,
    'Insert BatchNorm1d between the first Linear and ReLU of the projection '
    'head. Mitigates dimensional collapse. Default off (matches SupCon/SimCLR '
    'reference); turn on if features collapse (loss sits at log(2B-1)).')
FLAGS = absl.flags.FLAGS


# Per-dataset validation split size used by paper/utils/datasets.py. We need
# to match this exactly because split_dataset first carves off `val_size`
# samples and only then takes the train_size subset from the remainder — so
# any mismatch here would shift which indices end up in the training subset.
_VAL_SIZE = {'CIFAR10': 6000, 'SVHN': 6000, 'CINIC10': 10}


# Per-dataset specs: torchvision dataset class, its train-split kwargs,
# per-channel normalization stats (must match paper/utils/datasets.py so
# downstream Memory Wrap training sees features on the same scale), and
# whether horizontal flip is an identity-preserving augmentation.
DATASET_SPECS = {
    'CIFAR10': {
        'cls': datasets.CIFAR10,
        'split_kwargs': {'train': True},
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.2023, 0.1994, 0.2010],
        'hflip': True,   # cats/planes/etc. are roughly symmetric
    },
    'SVHN': {
        'cls': datasets.SVHN,
        'split_kwargs': {'split': 'train'},
        'mean': [0.485, 0.456, 0.406],   # matches paper/utils/datasets.py get_SVHN
        'std':  [0.229, 0.224, 0.225],
        'hflip': False,  # '3' flipped is not a '3'
    },
    'CINIC10': {
        # CINIC-10 is distributed as plain image folders (not a torchvision
        # dataset class), so we load it via datasets.ImageFolder rooted at
        # <data_dir>/CINIC10/train. The user must download/extract the
        # tarball manually before running this script (see plan Step 1).
        'imagefolder_subdir': 'CINIC10/train',
        'mean': [0.47889522, 0.47227842, 0.43047404],  # matches paper/utils/datasets.py get_CINIC10
        'std':  [0.24205776, 0.23828046, 0.25874835],
        'hflip': True,   # CIFAR-10-style natural images, symmetric
    },
}


def contrastive_loss(features, labels, temp=0.07):
    """SupCon (Khosla et al., 2020) or SimCLR (Chen et al., 2020) loss.

    Both losses share the same softmax-over-similarities structure. The only
    difference is WHICH pairs count as positives:
      - SupCon: all same-class feature pairs (uses labels).
      - SimCLR: only the two-view pair of the same image (labels=None).

    Args:
        features: [2B, d] L2-normalized feature vectors. First B rows are
            view-1 of each image, next B rows are view-2 of the same images
            (in the same order).
        labels:   [B] class labels (SupCon mode) or None (SimCLR mode).
        temp:     softmax temperature; lower = harder negatives dominate.

    Returns:
        Scalar loss averaged over all 2B anchors.
    """
    twoB = features.size(0)
    B = twoB // 2

    # Positive mask: mask[i, j] = 1 iff feature j is a positive for anchor i.
    # Shape: [2B, 2B].
    if labels is None:
        # SimCLR: the only positive for anchor i is its OTHER augmented view
        # (same underlying image). Since rows 0..B-1 are view-1 and rows
        # B..2B-1 are view-2 of the same images in the same order, the
        # desired mask is the identity rolled by B columns — this places a
        # 1 at position (i, (i+B) % 2B) for every i.
        mask = torch.eye(twoB, device=features.device).roll(B, dims=1)
    else:
        # SupCon: duplicate labels (view-1 and view-2 share their class),
        # then pairs with matching labels become positives. Zero the
        # diagonal so an anchor is never its own positive.
        labels = torch.cat([labels, labels])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

    # Pairwise cosine similarities scaled by temperature. Because features
    # are L2-normalized, matmul is equivalent to pairwise cosine. Shape:
    # [2B, 2B]. logits[i, j] = sim(anchor_i, sample_j) / temp.
    logits = features @ features.T / temp

    # Numerical stability: subtract per-row max from each logit before
    # exponentiating. This doesn't change the softmax (constants cancel)
    # but keeps exp() from overflowing in fp16 / large-temp regimes.
    # .detach() so the subtracted max isn't part of the backward graph.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Denominator for log-softmax must EXCLUDE the self-similarity (which is
    # always 1/temp after normalization — trivially the largest logit and
    # would dominate the softmax). `not_self` is a [2B, 2B] matrix with 1s
    # everywhere except the diagonal.
    not_self = 1 - torch.eye(twoB, device=features.device)

    # log_prob[i, j] = log P(sample j | anchor i) under the softmax over all
    # non-self samples. The 1e-12 is a guard against log(0) if the entire
    # row of exponentials happens to be zero (effectively never, but safe).
    log_prob = logits - torch.log((logits.exp() * not_self).sum(dim=1, keepdim=True) + 1e-12)

    # For each anchor i: average log_prob over its positives P(i).
    #   - (mask * log_prob).sum(dim=1): sum of log-probs over positive j.
    #   - mask.sum(dim=1): |P(i)|, the number of positives for anchor i.
    #   - .clamp(min=1): edge case — if an anchor happens to have no
    #     positives in this batch (rare with balanced sampling) divide by 1
    #     instead of 0. Since the numerator is also 0 for such anchors,
    #     their contribution to the mean becomes 0, not NaN.
    # Final: negate (SupCon maximizes log-prob so loss minimizes -log-prob)
    # and average over all 2B anchors.
    return -(mask * log_prob).sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()


class TwoViews:
    """Return two independently-augmented versions of the same input image.

    Wrapping the torchvision transform in this class is the standard SimCLR /
    SupCon trick: it makes DataLoader yield batches shaped as
        ((view1_batch, view2_batch), label_batch)
    so we get two stochastic views of every image to use as a known positive
    pair in the contrastive loss.
    """
    def __init__(self, t): self.t = t
    def __call__(self, x): return (self.t(x), self.t(x))


def main(argv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Input shape is fixed (32x32, constant batch size thanks to drop_last),
    # so let cuDNN benchmark kernels at startup and pick the fastest for
    # each conv. Free ~5-15% speedup on conv-heavy backbones.
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(FLAGS.seed)
    spec = DATASET_SPECS[FLAGS.dataset]
    budget_dir = 'full' if FLAGS.train_examples == 0 else str(FLAGS.train_examples)
    tracker = tracking.init(
        name=f'{FLAGS.loss}-seed{FLAGS.seed}',
        group=f'{FLAGS.dataset}-{budget_dir}-{FLAGS.loss}-pretrain',
        job_type='pretrain',
        config={f.name: f.value for f in FLAGS.get_flags_for_module(sys.argv[0])})

    # --- Augmentation pipeline ----------------------------------------------
    # SimCLR-style augmentations: strong enough that two views of the same
    # image look meaningfully different, but not so strong that class
    # content is destroyed. Shared with train.py --augment.
    aug = contrastive_augmentation(spec['mean'], spec['std'], spec['hflip'])

    # Dataset returns ((view1, view2), label) per sample thanks to TwoViews.
    # CINIC-10 uses ImageFolder (no torchvision auto-download); CIFAR-10/SVHN
    # use their torchvision dataset class with download=True.
    if 'imagefolder_subdir' in spec:
        ds = datasets.ImageFolder(os.path.join(FLAGS.data_dir, spec['imagefolder_subdir']),
                                  transform=TwoViews(aug))
    else:
        ds = spec['cls'](FLAGS.data_dir, download=True,
                         transform=TwoViews(aug), **spec['split_kwargs'])
    # Optional same-data-budget subsetting. When --train_examples > 0 we
    # reproduce downstream's seeded split EXACTLY so the pretrain subset is
    # the same image indices that train.py will hand to Memory Wrap. We
    # discard the val subset because contrastive pretraining doesn't
    # validate — the loss isn't an accuracy proxy.
    if FLAGS.train_examples > 0:
        ds, _ = split_dataset(ds, FLAGS.train_examples,
                              _VAL_SIZE[FLAGS.dataset], FLAGS.seed)
        print(f'Subsetting to {len(ds)} samples (seed={FLAGS.seed}, '
              f'val_size={_VAL_SIZE[FLAGS.dataset]}) — matches downstream '
              f"train.py with train_examples={FLAGS.train_examples}.")
    # drop_last=True: SupCon needs a predictable 2B batch shape; dropping
    # the incomplete final batch avoids per-epoch shape edge cases.
    # persistent_workers=True: don't tear down and respawn worker processes
    #   between epochs (saves ~1-2s of Python startup per epoch).
    # prefetch_factor=4: each worker keeps 4 batches queued ahead of the
    #   GPU, hiding CPU augmentation latency behind GPU compute.
    loader = torch.utils.data.DataLoader(ds, batch_size=FLAGS.batch_size,
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=FLAGS.num_workers, persistent_workers=FLAGS.num_workers > 0,
        prefetch_factor=4 if FLAGS.num_workers > 0 else None)

    # --- Model --------------------------------------------------------------
    # We instantiate the 'encoder_memory' variant (= real Memory Wrap) so we
    # get access to `forward_encoder`, which returns the [B, d] feature
    # vector that Memory Wrap attends over. The self.mw head is PRESENT on
    # the model but never called during pretraining — it stays at random
    # init and receives no gradient updates, so its parameters persist
    # untouched into the saved checkpoint. Downstream train.py skips those
    # keys and keeps its own freshly initialized head.
    model = utils.get_model(FLAGS.model, 10, model_type='encoder_memory').to(device)

    # --- Projection head ----------------------------------------------------
    # Standard SupCon/SimCLR recipe: contrastive loss is applied to a small
    # MLP on top of the encoder, not to the encoder output directly. The MLP
    # absorbs augmentation-invariance pressure so encoder features stay rich
    # for downstream use. Discarded at save time.
    #
    # We probe the encoder's output dim with a dummy forward pass so the head
    # adapts to any backbone (mobilenet=1280, resnet18=512, googlenet=1024,
    # densenet=342, ...). Done in eval() to avoid touching BN running stats.
    if FLAGS.projection_dim > 0:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 32, 32, device=device)
            enc_dim = model.forward_encoder(dummy).shape[1]
        model.train()
        # 2-layer MLP: (enc_dim -> enc_dim -> projection_dim). Hidden width
        # equal to encoder dim follows the SupCon paper's recipe. Optional
        # BatchNorm1d between the first Linear and ReLU (see --projection_bn):
        # not in the SupCon reference, but present in BYOL/MoCo v3/DINO and
        # mitigates dimensional collapse when the encoder outputs have low
        # between-sample variance at init.
        layers = [torch.nn.Linear(enc_dim, enc_dim)]
        if FLAGS.projection_bn:
            layers.append(torch.nn.BatchNorm1d(enc_dim))
        layers += [
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(enc_dim, FLAGS.projection_dim),
        ]
        projection = torch.nn.Sequential(*layers).to(device)
        bn_tag = ' (with BatchNorm1d)' if FLAGS.projection_bn else ''
        print(f'Projection head{bn_tag}: {enc_dim} -> {enc_dim} -> '
              f'{FLAGS.projection_dim} (pretraining only; discarded at save).',
              flush=True)
    else:
        # Legacy / ablation: identity head. Loss applied directly to encoder.
        projection = torch.nn.Identity().to(device)
        print('Projection head: DISABLED (--projection_dim=0). '
              'Loss applied directly to encoder features.', flush=True)

    # SGD + momentum + weight decay matches the SupCon paper's recipe for
    # CIFAR-10. Nesterov gives a small convergence boost on this setup. The
    # projection head must be optimized jointly with the encoder; for
    # nn.Identity this is a no-op (no parameters).
    opt = torch.optim.SGD(
        list(model.parameters()) + list(projection.parameters()),
        lr=FLAGS.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # LR schedule: linear warmup (epochs 0..warmup_epochs) followed by cosine
    # decay (warmup_epochs..epochs). Warmup starts at lr * 1e-2 to avoid a
    # hard zero (which would stall SGD+momentum). Cosine phase anneals from
    # FLAGS.lr to 0 over the remaining epochs.
    #
    # Why warmup matters here: the freshly-initialized projection head has
    # no prior and can't survive LR=0.5 on the first few steps. Without
    # warmup, projection-head runs collapse (loss = log(2B-1), all features
    # identical, gradients vanish). With warmup, the head gently enters the
    # useful regime before the full LR is applied. See --warmup_epochs.
    if FLAGS.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-2, end_factor=1.0,
            total_iters=FLAGS.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, FLAGS.epochs - FLAGS.warmup_epochs))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[FLAGS.warmup_epochs])
        print(f'LR schedule: linear warmup over {FLAGS.warmup_epochs} epochs '
              f'(lr scales from {FLAGS.lr*1e-2:.4f} to {FLAGS.lr}), then '
              f'cosine decay over the remaining '
              f'{FLAGS.epochs - FLAGS.warmup_epochs} epochs.', flush=True)
    else:
        # Legacy: plain cosine from epoch 0. Works without a projection head
        # but will collapse the projection head run at default lr.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FLAGS.epochs)
        print('LR schedule: cosine decay (no warmup). WARNING: collapse-prone '
              'when --projection_dim > 0 at high LR.', flush=True)
    # Automatic mixed precision: roughly 2x training speedup on modern GPUs
    # with negligible accuracy impact. GradScaler handles the dynamic loss
    # scaling needed to prevent fp16 gradient underflow.
    scaler = torch.cuda.amp.GradScaler()

    # --- Training loop ------------------------------------------------------
    model.train()
    # One-time diagnostic helper: after the first forward pass of epoch 1, we
    # print the feature statistics (mean pairwise cosine sim, feature std per
    # dim) so collapse is immediately visible in the logs. If `mean_cos` is
    # ~1.0 and per-dim std is ~0 at epoch 1, features are collapsed and no
    # amount of additional epochs will fix it --- you need --projection_bn or
    # a different LR.
    printed_diag = False
    for ep in range(1, FLAGS.epochs + 1):
        epoch_loss = 0.0
        for (v1, v2), y in loader:
            # Stack both views into a single tensor of shape [2B, 3, 32, 32].
            # Encoding both views in the same forward pass keeps BatchNorm
            # statistics consistent across views (important! — separate
            # forward passes would compute different running means/stds for
            # the two views and degrade the contrastive signal).
            imgs = torch.cat([v1, v2]).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                # forward_encoder returns [2B, enc_dim] raw features. The
                # projection head (Identity if disabled) maps these to
                # [2B, projection_dim]. F.normalize then projects onto the
                # unit hypersphere so matmul later = cosine similarity.
                feat = F.normalize(projection(model.forward_encoder(imgs)), dim=1)
                # SupCon: pass labels. SimCLR: pass None. Hybrid: both,
                # combined as alpha * supcon + (1 - alpha) * simclr.
                if FLAGS.loss == 'supcon':
                    loss = contrastive_loss(feat, y, FLAGS.temperature)
                elif FLAGS.loss == 'simclr':
                    loss = contrastive_loss(feat, None, FLAGS.temperature)
                else:  # hybrid
                    l_sup = contrastive_loss(feat, y, FLAGS.temperature)
                    l_sim = contrastive_loss(feat, None, FLAGS.temperature)
                    loss = FLAGS.hybrid_alpha * l_sup + (1 - FLAGS.hybrid_alpha) * l_sim

            # One-shot diagnostic: first batch of epoch 1 only. Runs after the
            # autocast block using the already-computed `feat` (cast to fp32
            # for numerically stable stats). Distinguishes "features are fine
            # but loss is high" (expected early; mean_cos near 0, per_dim_std
            # non-trivial) from "features are already collapsed" (mean_cos
            # near 1 and/or per_dim_std near 0; needs --projection_bn or a
            # different LR / init).
            if not printed_diag:
                with torch.no_grad():
                    f32 = feat.float()
                    sim = f32 @ f32.T
                    sim.fill_diagonal_(0)
                    n = f32.size(0)
                    mean_cos = sim.sum().item() / (n * (n - 1))
                    per_dim_std = f32.std(dim=0).mean().item()
                print(f'[diag ep1/batch1] mean_cos(off-diag)={mean_cos:.4f}  '
                      f'per_dim_std(mean)={per_dim_std:.4f}  '
                      f'(mean_cos ~ 1.0 OR per_dim_std ~ 0 => features '
                      f'collapsed; try --projection_bn)', flush=True)
                tracker.summary['diag/mean_cos'] = mean_cos
                tracker.summary['diag/per_dim_std'] = per_dim_std
                printed_diag = True

            # scaler.scale: multiplies loss by dynamic scale factor to keep
            # fp16 gradients in representable range.
            scaler.scale(loss).backward()
            # scaler.step: unscales grads and calls optimizer.step(), but
            # skips the step if inf/NaN gradients are detected.
            scaler.step(opt)
            # scaler.update: adjusts the scale factor for next iteration.
            scaler.update()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        lr = opt.param_groups[0]['lr']
        sched.step()  # Cosine schedule steps once per epoch, not per batch.
        # flush=True: when stdout is redirected to a log file Python block-
        # buffers to ~4KB, which with only ~30 chars per print would never
        # flush during a 100-epoch run. Explicit flush keeps logs live.
        print(f'Epoch {ep}/{FLAGS.epochs}  loss={mean_loss:.4f}  lr={lr:.4g}', flush=True)
        tracker.log({'pretrain/loss': mean_loss, 'pretrain/lr': lr, 'epoch': ep})

    # --- Save checkpoint ----------------------------------------------------
    # Checkpoint format mirrors what train.py saves: model_name and
    # dataset_name are used by downstream scripts (generate_memory_images.py,
    # generate_heatmaps.py) to reconstruct the correct architecture. The
    # 'modality' key is informational — train.py reads state_dict only.
    # Save path includes both dataset and loss so runs don't clobber each
    # other when pilot-comparing across datasets or objectives.
    # Include the data budget and seed in the save path so 'full' and
    # budgeted runs, and the encoders for different seeds, live side by side.
    out = f'models/{FLAGS.dataset}/{FLAGS.loss}/{FLAGS.model}/{budget_dir}/seed{FLAGS.seed}.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # train.py checks dataset_name, train_examples and seed against the run
    # before loading, so an encoder can never be used with the wrong subset.
    torch.save({'model_state_dict': model.state_dict(), 'model_name': FLAGS.model,
                'num_classes': 10, 'modality': f'{FLAGS.loss}_pretrained',
                'dataset_name': FLAGS.dataset,
                'train_examples': FLAGS.train_examples,
                'seed': FLAGS.seed}, out)
    print(f'Saved {out}')
    tracker.finish()


if __name__ == '__main__':
    absl.app.run(main)
