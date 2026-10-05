"""Run plan/svhn_eval_plan.md stages A-E end to end (SVHN-500, MobileNetV2).

A: hyperparameter search on 100 held-out training images (3 seeds).
B: budget grid (train to plateau) and downstream search, same validation data.
C: final runs on test with all 500 images (10 seeds).
D: retrieval purity of the Memory Wrap cells.
E: statistics and plan/svhn_eval_results.md.

Every choice is made by the rules below from validation logs and written to
the results file before stage C starts. The driver is resumable: a job whose
log already holds all its runs (or whose encoder exists) is skipped, so the
same logs always give the same choices.

Set SLACK_WEBHOOK to get a message at the end of each stage and each final cell.

Usage (from paper/):
    SLACK_WEBHOOK=... python -u scripts/svhn_eval.py --jobs=4
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import parse_log, summarize  # noqa: E402

LOG_DIR = 'logs/svhn_eval'
RESULTS = '../plan/svhn_eval_results.md'
STATE = os.path.join(LOG_DIR, 'state.json')
PY = sys.executable

VAL = 100
SEARCH_SEEDS = 3
FINAL_SEEDS = 10
SCRATCH_GRID = [(lr, wd) for lr in (0.1, 0.03, 0.01) for wd in (5e-4, 5e-3)]
SUPCON_GRID = [(lr, t) for lr in (0.125, 0.25) for t in (0.07, 0.1, 0.2)]
REF_BUDGET = 80          # scratch search budget
SUPCON_REF_BUDGET = 40   # SupCon pretraining search budget
BUDGETS = [40, 80, 160]
PLATEAU_PP = 1.0
DOWN_WD = 5e-4
PROBE = (0.1, 40)  # frozen linear probe that scores SupCon pretraining (lr, epochs)
FORCE_EXTRA = False
DOWN_GRID = {'ULfz': [(0.1, 40), (0.01, 40)],
             'UMfz': [(0.1, 40), (0.01, 40)],
             'UMft': [(lr, ep) for lr in (0.1, 0.01) for ep in (40, 80)]}
CELLS = ['SL', 'SM', 'ULfz', 'UMfz', 'UMft']
CELL_NAMES = {'SL': 'Scratch + Linear', 'SM': 'Scratch + MW', 'ULfz': 'SupCon -> Linear, frozen',
              'UMfz': 'SupCon -> MW, frozen', 'UMft': 'SupCon -> MW, fine-tuned'}
COMMON = ['--set', 'train_examples=500', '--set', 'batch_size_train=64',
          '--set', 'dataset_name=SVHN', '--wandb']

PRETRAIN_WORKERS = 6
PURITY_EXTRA = []


def use_smoke_settings():
    """Tiny grids, budgets and seeds that exercise every stage in minutes."""
    global SEARCH_SEEDS, FINAL_SEEDS, SCRATCH_GRID, SUPCON_GRID, REF_BUDGET, SUPCON_REF_BUDGET
    global BUDGETS, DOWN_GRID, PURITY_EXTRA, PROBE, FORCE_EXTRA
    SEARCH_SEEDS, FINAL_SEEDS = 2, 2
    SCRATCH_GRID = [(0.1, 5e-4), (0.03, 5e-4)]
    SUPCON_GRID = [(0.125, 0.07), (0.25, 0.1)]
    REF_BUDGET, SUPCON_REF_BUDGET, BUDGETS = 3, 2, [2, 3, 4]
    DOWN_GRID = {'ULfz': [(0.1, 2)], 'UMfz': [(0.1, 2), (0.01, 2)], 'UMft': [(0.1, 2), (0.1, 3)]}
    PURITY_EXTRA = ['--max_images=300']
    PROBE, FORCE_EXTRA = (0.1, 2), True
_print_lock = threading.Lock()
_state_lock = threading.RLock()


def log(msg):
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    with _print_lock:
        print(f'[{stamp}] {msg}', flush=True)


def slack(text):
    url = os.environ.get('SLACK_WEBHOOK')
    log(f'SLACK: {text}')
    if not url:
        return
    try:
        req = urllib.request.Request(url, data=json.dumps({'text': text}).encode(),
                                     headers={'Content-type': 'application/json'})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:  # a failed ping must never stop the experiment
        log(f'slack failed: {e}')


def fmt(x):
    """Short, path-safe number: 0.0005 -> 5e-4 style kept readable."""
    return f'{x:g}'


# --- jobs -----------------------------------------------------------------

def run_cmd(name, args, done):
    """Run args with output to LOG_DIR/name.txt unless done(log) already holds.
    Retries once, then raises."""
    path = os.path.join(LOG_DIR, f'{name}.txt')
    if os.path.exists(path) and done(path):
        log(f'skip {name} (complete)')
        return path
    for attempt in (1, 2):
        log(f'start {name}' + (' (retry)' if attempt == 2 else ''))
        t0 = time.time()
        with open(path, 'w') as f:
            f.write('# ' + ' '.join(args) + '\n')
            f.flush()
            code = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT)
        if code == 0 and done(path):
            log(f'done {name} ({(time.time() - t0) / 60:.1f} min)')
            return path
        log(f'FAILED {name} (exit {code}), see {path}')
    raise RuntimeError(f'{name} failed twice; log: {path}')


def train(name, modality, runs, val, epochs, lr, wd, encoder=None, freeze=False, save=False):
    """train.py job. Returns per-seed accuracies in seed order."""
    args = [PY, '-u', 'train.py', f'--modality={modality}', f'--val_examples={val}',
            f'--tag={name}', '--set', f'runs={runs}', '--set', f'SVHN.num_epochs={epochs}',
            '--set', f'optimizer.learning_rate={lr}', '--set', f'optimizer.weight_decay={wd}',
            '--set', f'save={save}'] + COMMON
    if encoder:
        args.append(f'--pretrained_encoder={encoder}')
    if freeze:
        args.append('--freeze_encoder=True')
    path = run_cmd(name, args, lambda p: len(parse_log(p)) == runs)
    accs = parse_log(path)
    return [accs[k] for k in sorted(accs)]


def enc_tag(lr, t, epochs, val):
    return f'lr{fmt(lr)}_t{fmt(t)}_ep{epochs}_' + (f'val{val}' if val else 'final')


def enc_dir(tag):
    return f'models/SVHN/supcon/mobilenet/500_{tag}'


def pretrain(lr, t, epochs, val, seed):
    tag = enc_tag(lr, t, epochs, val)
    out = os.path.join(enc_dir(tag), f'seed{seed}.pt')
    if os.path.isfile(out):
        log(f'skip pretrain {tag} seed{seed} (exists)')
        return
    args = [PY, '-u', 'pretrain_supcon.py', '--dataset=SVHN', '--loss=supcon', '--model=mobilenet',
            '--train_examples=500', f'--val_examples={val}', f'--epochs={epochs}', '--batch_size=64',
            f'--lr={lr}', f'--temperature={t}', '--projection_dim=0',
            f'--num_workers={PRETRAIN_WORKERS}', f'--seed={seed}', f'--tag={tag}', '--wandb']
    run_cmd(f'pre_{tag}_seed{seed}', args, lambda p: os.path.isfile(out))


def scratch_name(head, lr, wd, epochs, val=VAL):
    return f'{head}_lr{fmt(lr)}_wd{fmt(wd)}_ep{epochs}_' + (f'val{val}' if val else 'final')


def down_name(cell, tag, lr, epochs):
    return f'{cell}_{tag}_dlr{fmt(lr)}_ep{epochs}'


def scratch_job(head, lr, wd, epochs, val=VAL, runs=None, save=False):
    runs = runs or SEARCH_SEEDS
    modality = 'std' if head == 'SL' else 'encoder_memory'
    return train(scratch_name(head, lr, wd, epochs, val), modality, runs, val, epochs, lr, wd, save=save)


def down_job(cell, tag, lr, epochs, val=VAL, runs=None, save=False):
    runs = runs or SEARCH_SEEDS
    modality = 'std' if cell == 'ULfz' else 'encoder_memory'
    return train(down_name(cell, tag, lr, epochs), modality, runs, val, epochs, lr, DOWN_WD,
                 encoder=enc_dir(tag), freeze=cell != 'UMft', save=save)


def supcon_probe(lr, t, epochs):
    """Pretrain the search seeds, then score them with a frozen linear probe."""
    for seed in range(SEARCH_SEEDS):
        pretrain(lr, t, epochs, VAL, seed)
    return down_job('ULfz', enc_tag(lr, t, epochs, VAL), *PROBE)


def model_dir(cell, name):
    """Where train.py saves the checkpoints of a run (see its save-path suffix)."""
    modality = 'std' if cell in ('SL', 'ULfz') else 'encoder_memory'
    suffix = '' if cell in ('SL', 'SM') else '_supcon'
    if cell in ('ULfz', 'UMfz'):
        suffix += '_frozen'
    return f'models/SVHN/{modality}{suffix}_{name}/mobilenet/500'


# --- selection rules --------------------------------------------------------

def best(results):
    """Key with the highest mean; ties go to the earlier grid entry."""
    return max(results, key=lambda k: (round(float(np.mean(results[k])), 6), -list(results).index(k)))


def plateau(by_budget):
    """Smallest budget whose mean is within PLATEAU_PP of the best budget's mean."""
    top = max(np.mean(v) for v in by_budget.values())
    return min(b for b, v in by_budget.items() if np.mean(v) >= top - PLATEAU_PP)


def needs_320(by_budget):
    return FORCE_EXTRA or np.mean(by_budget[BUDGETS[-1]]) - np.mean(by_budget[BUDGETS[-2]]) > PLATEAU_PP


def ms(accs):
    m, s, _, n = summarize(accs)
    return f'{m:.2f} ± {s:.2f} (n={n})'


# --- results file -------------------------------------------------------------

def save_state(state):
    with _state_lock:
        with open(STATE, 'w') as f:
            json.dump(state, f, indent=1)


def commit(state, **updates):
    """Apply top-level or nested updates under the lock, then save and render.
    Worker threads share state, so every write goes through here."""
    with _state_lock:
        for key, value in updates.items():
            *parents, leaf = key.split('/')
            node = state
            for parent in parents:
                node = node.setdefault(parent, {})
            node[leaf] = value
        save_state(state)
        render(state)


def grid_table(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


def render(state):
    out = ['# SVHN-500 evaluation results', '',
           'Generated by `paper/scripts/svhn_eval.py` from `paper/logs/svhn_eval/`. '
           'Validation = 100 images held out of each seed\'s 500; search seeds 0-2; final runs seeds 0-9 on test. '
           'Accuracies are mean ± sample std over seeds.', '']
    a = state.get('A')
    if a:
        out += ['## Stage A: hyperparameter search (validation)', '']
        for head in ('SL', 'SM'):
            rows = [(fmt(r['lr']), fmt(r['wd']), ms(r['accs'])) for r in a[head]['grid']]
            out += [f'### {CELL_NAMES[head]}, {REF_BUDGET} epochs', '', grid_table(rows, ['lr', 'wd', 'val acc']), '',
                    f'Chosen: lr {fmt(a[head]["lr"])}, wd {fmt(a[head]["wd"])}.', '']
        rows = [(fmt(r['lr']), fmt(r['t']), ms(r['accs'])) for r in a['supcon']['grid']]
        out += [f'### SupCon pretraining, {SUPCON_REF_BUDGET} epochs (frozen linear probe, lr {fmt(PROBE[0])}, {PROBE[1]} epochs)', '',
                grid_table(rows, ['lr', 'temperature', 'probe val acc']), '',
                f'Chosen: lr {fmt(a["supcon"]["lr"])}, temperature {fmt(a["supcon"]["t"])}.', '']
    b = state.get('B')
    if b:
        out += ['## Stage B: budgets and downstream (validation)', '',
                f'Plateau rule: smallest budget within {PLATEAU_PP} pp of the best budget\'s mean; '
                f'320 epochs only if 80 -> 160 gains more than {PLATEAU_PP} pp.', '']
        rows = []
        for key, label in (('SL', 'Scratch + Linear'), ('SM', 'Scratch + MW'), ('supcon', 'SupCon pretraining (probe)')):
            for budget, accs in sorted(b[key]['budgets'].items(), key=lambda kv: int(kv[0])):
                rows.append((label, budget, ms(accs), '**chosen**' if int(budget) == b[key]['plateau'] else ''))
        out += [grid_table(rows, ['method', 'epochs', 'val acc', '']), '']
        rows = [(CELL_NAMES[c], fmt(r['lr']), r['epochs'], ms(r['accs']),
                 '**chosen**' if (r['lr'], r['epochs']) == (b['down'][c]['lr'], b['down'][c]['epochs']) else '')
                for c in ('ULfz', 'UMfz', 'UMft') for r in b['down'][c]['grid']]
        out += [f'### Downstream after SupCon (pretraining {b["supcon"]["plateau"]} epochs)', '',
                grid_table(rows, ['cell', 'lr', 'epochs', 'val acc', '']), '']
        out += ['### Final settings (fixed before any test run)', '']
        rows = [(CELL_NAMES.get(c, 'SupCon pretraining'), s['desc']) for c, s in state['final_settings'].items()]
        out += [grid_table(rows, ['cell', 'settings']), '']
    c = state.get('C')
    if c:
        out += ['## Stage C: final runs (test, all 500 images)', '']
        rows = []
        for cell in CELLS:
            if cell in c:
                m, s, h, n = summarize(c[cell])
                rows.append((CELL_NAMES[cell], f'{m:.2f}', f'{s:.2f}', f'± {h:.2f}', n,
                             ' '.join(f'{x:.1f}' for x in c[cell])))
        out += [grid_table(rows, ['cell', 'mean', 'sample std', '95% CI', 'n', 'per seed']), '']
    if state.get('tests'):
        out += ['### Welch t-tests (Holm-corrected across the family)', '',
                grid_table([(t['label'], f'{t["diff"]:+.2f}', f'{t["p"]:.4f}', f'{t["p_holm"]:.4f}')
                            for t in state['tests']], ['comparison', 'difference (pp)', 'p', 'p (Holm)']), '']
        q3 = state['interaction']
        out += [f'Q3 interaction, (UMfz - ULfz) - (SM - SL): {q3["est"]:+.2f} pp, '
                f'approx. 95% CI [{q3["lo"]:+.2f}, {q3["hi"]:+.2f}].', '']
    if state.get('figure'):
        rows = [(CELL_NAMES[h], e, ms(v)) for h, d in state['figure'].items()
                for e, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
        out += ['### Scratch on test at every budget (figure only; choices came from stage B)', '',
                grid_table(rows, ['cell', 'epochs', 'test acc']), '']
    if state.get('D'):
        out += ['## Stage D: purity (test, 5 memory redraws)', '']
        out += [f'- **{k}**: `{v}`' for k, v in state['D'].items()]
        out += ['']
    if state.get('plot'):
        out += [f'Budget plot: `{state["plot"]}`', '']
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        f.write('\n'.join(out) + '\n')


# --- stages -------------------------------------------------------------------

def stage_a(pool, state):
    futs = {('SL', g): pool.submit(scratch_job, 'SL', *g, REF_BUDGET) for g in SCRATCH_GRID}
    futs.update({('SM', g): pool.submit(scratch_job, 'SM', *g, REF_BUDGET) for g in SCRATCH_GRID})
    futs.update({('supcon', g): pool.submit(supcon_probe, *g, SUPCON_REF_BUDGET) for g in SUPCON_GRID})
    res = {k: f.result() for k, f in futs.items()}
    a = {}
    for head in ('SL', 'SM'):
        grid = {g: res[(head, g)] for g in SCRATCH_GRID}
        lr, wd = best(grid)
        a[head] = {'lr': lr, 'wd': wd, 'grid': [{'lr': g[0], 'wd': g[1], 'accs': v} for g, v in grid.items()]}
    grid = {g: res[('supcon', g)] for g in SUPCON_GRID}
    lr, t = best(grid)
    a['supcon'] = {'lr': lr, 't': t, 'grid': [{'lr': g[0], 't': g[1], 'accs': v} for g, v in grid.items()]}
    commit(state, A=a)
    slack('SVHN eval stage A done (validation search). '
          f'Scratch+Linear: lr {fmt(a["SL"]["lr"])} wd {fmt(a["SL"]["wd"])} -> {ms(grid_mean(a, "SL"))}. '
          f'Scratch+MW: lr {fmt(a["SM"]["lr"])} wd {fmt(a["SM"]["wd"])} -> {ms(grid_mean(a, "SM"))}. '
          f'SupCon: lr {fmt(lr)} temp {fmt(t)} -> probe {ms(grid[(lr, t)])}.')


def grid_mean(a, head):
    return next(r['accs'] for r in a[head]['grid'] if (r['lr'], r['wd']) == (a[head]['lr'], a[head]['wd']))


def stage_b(pool, state):
    a = state['A']
    sl, sm, sc = a['SL'], a['SM'], a['supcon']

    def budget_runs(budgets):
        futs = {}
        for b in budgets:
            futs[('SL', b)] = pool.submit(scratch_job, 'SL', sl['lr'], sl['wd'], b)
            futs[('SM', b)] = pool.submit(scratch_job, 'SM', sm['lr'], sm['wd'], b)
            futs[('supcon', b)] = pool.submit(supcon_probe, sc['lr'], sc['t'], b)
        return {k: f.result() for k, f in futs.items()}

    res = budget_runs(BUDGETS)
    by = {m: {b: res[(m, b)] for b in BUDGETS} for m in ('SL', 'SM', 'supcon')}
    extra = [m for m in by if needs_320(by[m])]
    if extra:
        log(f'320 epochs needed for {extra}')
        futs = {}
        for m in extra:
            if m == 'supcon':
                futs[m] = pool.submit(supcon_probe, sc['lr'], sc['t'], 2 * BUDGETS[-1])
            else:
                cfg = a[m]
                futs[m] = pool.submit(scratch_job, m, cfg['lr'], cfg['wd'], 2 * BUDGETS[-1])
        for m, f in futs.items():
            by[m][2 * BUDGETS[-1]] = f.result()
    b = {m: {'budgets': {str(k): v for k, v in by[m].items()}, 'plateau': plateau(by[m])} for m in by}

    pre = b['supcon']['plateau']
    tag = enc_tag(sc['lr'], sc['t'], pre, VAL)
    futs = {(c, g): pool.submit(down_job, c, tag, *g) for c, grid in DOWN_GRID.items() for g in grid}
    res = {k: f.result() for k, f in futs.items()}
    b['down'] = {}
    for c, grid in DOWN_GRID.items():
        scores = {g: res[(c, g)] for g in grid}
        lr, ep = best(scores)
        b['down'][c] = {'lr': lr, 'epochs': ep, 'grid': [{'lr': g[0], 'epochs': g[1], 'accs': v} for g, v in scores.items()]}
    final = {}
    for head in ('SL', 'SM'):
        final[head] = {'lr': a[head]['lr'], 'wd': a[head]['wd'], 'epochs': b[head]['plateau']}
        final[head]['desc'] = f'end to end, lr {fmt(final[head]["lr"])}, wd {fmt(final[head]["wd"])}, {final[head]["epochs"]} epochs'
    final['pretrain'] = {'lr': sc['lr'], 't': sc['t'], 'epochs': pre,
                         'desc': f'SupCon lr {fmt(sc["lr"])}, temperature {fmt(sc["t"])}, {pre} epochs, batch 64, no projection head'}
    for c in ('ULfz', 'UMfz', 'UMft'):
        d = b['down'][c]
        final[c] = {'lr': d['lr'], 'epochs': d['epochs'],
                    'desc': f'pretraining as above, then {"frozen encoder (BN frozen)" if c != "UMft" else "fine-tuned"}, '
                            f'lr {fmt(d["lr"])}, wd {fmt(DOWN_WD)}, {d["epochs"]} epochs'}
    commit(state, B=b, final_settings=final)
    slack('SVHN eval stage B done; choices recorded before any test run. '
          f'Plateaus: Scratch+Linear {b["SL"]["plateau"]} ep, Scratch+MW {b["SM"]["plateau"]} ep, '
          f'SupCon pretraining {pre} ep' + (f' (320 run for {extra})' if extra else '') + '. Downstream: ' +
          ', '.join(f'{c} lr {fmt(b["down"][c]["lr"])} {b["down"][c]["epochs"]} ep' for c in ('ULfz', 'UMfz', 'UMft')) +
          '. Starting 10-seed test runs.')


def final_name(cell, fs):
    if cell in ('SL', 'SM'):
        return scratch_name(cell, fs[cell]['lr'], fs[cell]['wd'], fs[cell]['epochs'], val=0)
    p = fs['pretrain']
    return down_name(cell, enc_tag(p['lr'], p['t'], p['epochs'], 0), fs[cell]['lr'], fs[cell]['epochs'])


def stage_c(pool, state):
    fs = state['final_settings']
    p = fs['pretrain']

    def cell_done(cell, accs):
        commit(state, **{f'C/{cell}': accs})
        slack(f'SVHN eval final cell done: {CELL_NAMES[cell]} = {ms(accs)} test acc (10 seeds).')

    def scratch_final(cell):
        accs = scratch_job(cell, fs[cell]['lr'], fs[cell]['wd'], fs[cell]['epochs'], val=0, runs=FINAL_SEEDS, save=True)
        cell_done(cell, accs)

    def supcon_final(cell):
        tag = enc_tag(p['lr'], p['t'], p['epochs'], 0)
        accs = down_job(cell, tag, fs[cell]['lr'], fs[cell]['epochs'], val=0, runs=FINAL_SEEDS, save=True)
        cell_done(cell, accs)

    pre = [pool.submit(pretrain, p['lr'], p['t'], p['epochs'], 0, s) for s in range(FINAL_SEEDS)]
    futs = [pool.submit(scratch_final, c) for c in ('SL', 'SM')]
    for f in pre:
        f.result()
    futs += [pool.submit(supcon_final, c) for c in ('ULfz', 'UMfz', 'UMft')]
    for f in futs:
        f.result()


def purity(name, args):
    path = run_cmd(name, [PY, '-u', 'scripts/purity_score.py', '--dir_dataset=datasets'] + PURITY_EXTRA + args,
                   lambda p: any(k in open(p).read() for k in ('SUMMARY', 'Mean support-set size')))
    lines = [l.strip() for l in open(path, errors='replace') if l.startswith(('SUMMARY', 'Mean support-set size', 'Retrieved purity'))]
    return ' / '.join(lines)


def figure_runs(state, deadline):
    """Scratch cells on test at every budget, for a figure only (optional)."""
    fs = state['final_settings']
    for head in ('SL', 'SM'):
        for budget in sorted(int(k) for k in state['B'][head]['budgets']):
            if datetime.datetime.now() > deadline:
                log('skipping remaining figure runs (time)')
                return
            accs = scratch_job(head, fs[head]['lr'], fs[head]['wd'], budget, val=0, runs=FINAL_SEEDS, save=True)
            commit(state, **{f'figure/{head}/{budget}': accs})


def stage_d(pool, state, figure_deadline):
    fs = state['final_settings']
    dirs = {c: model_dir(c, final_name(c, fs)) for c in ('SM', 'UMfz', 'UMft')}
    jobs = {f'single {c}': ['--path=' + d, '--wandb'] for c, d in dirs.items()}
    for a_, b_ in (('UMft', 'SM'), ('UMfz', 'SM'), ('UMft', 'UMfz')):
        jobs[f'compare {a_} vs {b_}'] = ['--path=' + dirs[a_], '--compare_path=' + dirs[b_], '--wandb']
    for c, d in dirs.items():
        jobs[f'diagnose {c} run 1'] = ['--path=' + d + '/1.pt', '--max_images=2000', '--num_redraws=1', '--diagnose']
    fig = pool.submit(figure_runs, state, figure_deadline)
    futs = {k: pool.submit(purity, 'purity_' + k.replace(' ', '_'), v) for k, v in jobs.items()}
    commit(state, D={k: f.result() for k, f in futs.items()})
    slack('SVHN eval stage D (purity) done. ' + ' | '.join(
        f'{k}: {v.split(" / ")[0][:160]}' for k, v in state['D'].items() if not k.startswith('diagnose')))
    fig.result()


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def stage_e(state):
    c = state['C']
    pairs = [('Q1 Scratch MW vs Scratch Linear', 'SM', 'SL'),
             ('Q2 SupCon Linear frozen vs Scratch Linear', 'ULfz', 'SL'),
             ('Q2 SupCon MW frozen vs Scratch MW', 'UMfz', 'SM'),
             ('Q2 SupCon MW fine-tuned vs Scratch MW', 'UMft', 'SM'),
             ('Q3 SupCon MW frozen vs SupCon Linear frozen', 'UMfz', 'ULfz'),
             ('Q4 SupCon MW fine-tuned vs SupCon MW frozen', 'UMft', 'UMfz')]
    tests = []
    for label, x, y in pairs:
        r = stats.ttest_ind(c[x], c[y], equal_var=False)
        tests.append({'label': label, 'diff': float(np.mean(c[x]) - np.mean(c[y])), 'p': float(r.pvalue)})
    for t, adj in zip(tests, holm([t['p'] for t in tests])):
        t['p_holm'] = float(adj)
    est = (np.mean(c['UMfz']) - np.mean(c['ULfz'])) - (np.mean(c['SM']) - np.mean(c['SL']))
    se = np.sqrt(sum(np.var(c[k], ddof=1) / len(c[k]) for k in ('UMfz', 'ULfz', 'SM', 'SL')))
    commit(state, tests=tests, interaction={'est': float(est), 'lo': float(est - 1.96 * se), 'hi': float(est + 1.96 * se)})

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        all_x = set()
        for key, label in (('SL', 'Scratch + Linear'), ('SM', 'Scratch + MW'), ('supcon', 'SupCon pretraining (linear probe)')):
            bud = sorted(state['B'][key]['budgets'].items(), key=lambda kv: int(kv[0]))
            xs = [int(k) for k, _ in bud]
            all_x.update(xs)
            ax.errorbar(xs, [np.mean(v) for _, v in bud], yerr=[np.std(v, ddof=1) for _, v in bud],
                        marker='o', capsize=3, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(all_x))
        ax.set_xticklabels([str(x) for x in sorted(all_x)])
        ax.set_xlabel('epochs (SupCon: pretraining epochs)')
        ax.set_ylabel('validation accuracy (%)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot = '../plan/svhn_eval_budget.png'
        fig.savefig(plot, dpi=150)
        commit(state, plot='plan/svhn_eval_budget.png')
    except Exception as e:
        log(f'plot failed: {e}')
    summary = ', '.join(f'{k} {np.mean(v):.1f}' for k, v in c.items())
    slack(f'SVHN eval ALL DONE. Test acc: {summary}. Results in plan/svhn_eval_results.md on the pod.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4, help='concurrent jobs')
    parser.add_argument('--figure_deadline', default='06:00',
                        help='local time after which optional figure runs are skipped')
    parser.add_argument('--smoke', action='store_true', help='tiny settings to test the driver')
    args = parser.parse_args()
    if args.smoke:
        use_smoke_settings()
    os.makedirs(LOG_DIR, exist_ok=True)
    state = {}
    if os.path.exists(STATE):
        state = json.load(open(STATE))
    hh, mm = map(int, args.figure_deadline.split(':'))
    deadline = datetime.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= datetime.datetime.now():
        deadline += datetime.timedelta(days=1)
    log(f'optional figure runs stop at {deadline}')

    stage = 'setup'
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    try:
        for stage, fn in (('A', lambda: stage_a(pool, state)), ('B', lambda: stage_b(pool, state)),
                          ('C', lambda: stage_c(pool, state)), ('D', lambda: stage_d(pool, state, deadline)),
                          ('E', lambda: stage_e(state))):
            if stage == 'C' and len(state.get('C', {})) == len(CELLS):
                log('stage C complete, skipping')
                continue
            log(f'===== stage {stage} =====')
            t0 = time.time()
            fn()
            log(f'===== stage {stage} done in {(time.time() - t0) / 60:.1f} min =====')
    except Exception as e:
        traceback.print_exc()
        slack(f'SVHN eval FAILED in stage {stage}: {type(e).__name__}: {str(e)[:300]}')
        # Drop queued jobs; running ones finish so their logs stay complete.
        pool.shutdown(wait=True, cancel_futures=True)
        sys.exit(1)
    pool.shutdown()


if __name__ == '__main__':
    main()
