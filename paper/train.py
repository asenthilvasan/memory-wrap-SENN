
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import yaml
import utils.utils as utils
import utils.tracking as tracking
import time
import pickle
from typing import List

# user flags
torch.multiprocessing.set_sharing_strategy('file_system')

absl.flags.DEFINE_string("modality", None, "std, memory or encoder_memory")
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_integer("log_interval",100,"Log interval between prints during training process")
absl.flags.DEFINE_string("pretrained_encoder", None,
    "Optional directory of per-seed encoders from pretrain_supcon.py "
    "(e.g. models/SVHN/supcon/mobilenet/2000). Run N loads seedN.pt from it.")
absl.flags.DEFINE_bool("freeze_encoder", False,
    "If True, freeze all parameters except the classification head. "
    "Typically used together with --pretrained_encoder to do linear-probe style training.")
absl.flags.DEFINE_bool("augment", False,
    "Train on SupCon's pretraining augmentations (SVHN only). Controls for "
    "SupCon's augmentations when comparing against scratch.")
absl.flags.DEFINE_string("tag", None,
    "Optional suffix for the save directory and W&B group, e.g. 'ep80' for a "
    "run whose config/train.yaml was changed.")
absl.flags.DEFINE_integer("val_examples", 0,
    "Hold out this many of the train_examples images (SVHN only) and evaluate "
    "on them instead of the test set. 0 trains on all images and evaluates on test.")
absl.flags.DEFINE_multi_string("set", [],
    "Override a config/train.yaml key, e.g. --set SVHN.num_epochs=160 "
    "--set optimizer.learning_rate=0.03. Dotted keys reach nested values. "
    "Overriding SVHN.num_epochs alone rescales opt_milestones to 50%/75%.")
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS

# Parameter-name prefixes of the classification head across model variants
# (mw = Memory Wrap, linear = std MobileNetV2, the rest = other backbones).
HEAD_PREFIXES = ('mw.', 'linear.', 'classifier.', 'fc.', 'head.')


def load_pretrained_encoder(model:torch.nn.Module, encoder_dir:str, seed:int, config:dict):
    """Load the encoder that was pretrained on this run's training subset.

    Each run trains on a different seeded subset, so it must load the encoder
    pretrained with the same seed. Any other encoder has seen labels from a
    different subset, which gives the run extra labeled data.

    Raises:
        FileNotFoundError: if there is no encoder for this seed.
        ValueError: if the encoder was pretrained on a different subset.
        RuntimeError: if the encoder weights do not match the model.
    """
    path = os.path.join(encoder_dir, f'seed{seed}.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No pretrained encoder for seed {seed}: {path}')
    ckpt = torch.load(path, map_location='cpu')

    expected = {'dataset_name': config['dataset_name'],
                'train_examples': config['train_examples'],
                'val_examples': FLAGS.val_examples,
                'seed': seed}
    found = {key: ckpt.get(key) for key in expected}
    # Checkpoints from before validation holdouts trained on every image.
    found['val_examples'] = ckpt.get('val_examples', 0)
    if found != expected:
        raise ValueError(f'{path} was pretrained with {found}, but this run needs {expected}.')

    # Pretraining never trains the head, so keep this run's freshly initialized one.
    encoder_state = {k: v for k, v in ckpt['model_state_dict'].items()
                     if not k.startswith(HEAD_PREFIXES)}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    missing = [k for k in missing if not k.startswith(HEAD_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(f'Encoder weights in {path} do not match {type(model).__name__}. '
                           f'Missing: {missing}. Unexpected: {unexpected}.')
    print(f'Loaded pretrained encoder {path}', flush=True)


def set_train_mode(model:torch.nn.Module):
    """Put the model in training mode, keeping a frozen encoder's BatchNorm
    layers in eval mode so their running statistics stay as pretraining left
    them (a true linear probe). The head keeps training mode."""
    model.train()
    if not FLAGS.freeze_encoder:
        return
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not name.startswith(HEAD_PREFIXES):
            module.eval()


def apply_overrides(config:dict, overrides:List[str]) -> dict:
    """Apply --set key=value overrides to the YAML config in place.

    Values are parsed as YAML, so numbers and lists keep their types. When the
    epoch count is overridden without milestones, the milestones are rescaled
    to 50% and 75% of training (the default 40 epochs -> [20, 30]).
    """
    keys = set()
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'--set expects key=value, got {item!r}')
        *parents, leaf = key.split('.')
        node = config
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise KeyError(f'--set {key}: no such key in config/train.yaml')
        node[leaf] = yaml.safe_load(value)
        keys.add(key)
    dataset_name = config['dataset_name']
    if f'{dataset_name}.num_epochs' in keys and f'{dataset_name}.opt_milestones' not in keys:
        epochs = config[dataset_name]['num_epochs']
        config[dataset_name]['opt_milestones'] = [epochs // 2, 3 * epochs // 4]
    return config


def train_memory_model(model:torch.nn.Module,loaders:List[torch.utils.data.DataLoader],optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler,loss_criterion:torch.nn.modules.loss, num_epochs:int,device:torch.device,tracker=None)->torch.nn.Module:
    """ Function to train a model with a Memory Wrap layer (in the paper both
    the baseline variant and Memory Wrap)

    Args:
        model (torch.nn.Module): Model with a Memory Wrap layer to be trained
        loaders (List[torch.utils.data.DataLoader]): Loaders containing
            dataset subsets to be used to train the model. The loaders[0] 
            element is the training dataset, while loaders[1] contain the 
            dataset used to sample memory sets
        optimizer (torch.optim.Optimizer): PyTorch optimizer to use to perform
            training step
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate
        scheduler to adaptive adjusting the learning rate during training
        loss_criterion (torch.nn.modules.loss): criterion to use to compute
            the loss
        num_epochs (int): number of epoch to train the model
        device (torch.device): device where the model is stored
        tracker: W&B run (or no-op) that receives per-epoch metrics

    Returns:
        torch.nn.Module: the trained model
    """
    train_loader, mem_loader = loaders

    # training process
    set_train_mode(model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (data, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            # input
            data = data.to(device)
            y = y.to(device)
            memory_input, _ = next(iter(mem_loader))
            memory_input = memory_input.to(device)
            
            # perform training step
            with torch.cuda.amp.autocast():
                outputs  = model(data,memory_input)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()


            #log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')

        log_epoch('memory', epoch, num_epochs, epoch_loss / len(train_loader), optimizer, tracker)
        scheduler.step()# increase scheduler step for each epoch

    return model


def log_epoch(tag:str, epoch:int, num_epochs:int, mean_loss:float, optimizer:torch.optim.Optimizer, tracker):
    """Print a per-epoch heartbeat and send the same metrics to W&B."""
    lr = optimizer.param_groups[0]['lr']
    print(f'[{tag}] Epoch {epoch}/{num_epochs}  loss={mean_loss:.4f}  lr={lr:.4g}', flush=True)
    if tracker is not None:
        tracker.log({'train/loss': mean_loss, 'train/lr': lr, 'epoch': epoch})

def train_std_model(model:torch.nn.Module,train_loader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler, loss_criterion:torch.nn.modules.loss, num_epochs:int, device:torch.device=torch.device('cpu'),tracker=None)->torch.nn.Module:
    """ Function to train standard models

    Args:
        model (torch.nn.Module): standard PyTorch model
        train_loader (torch.utils.data.DataLoader): training dataset
        optimizer (torch.optim.Optimizer): PyTorch optimizer to use to perform
            training step
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate
        scheduler to adaptive adjusting the learning rate during training
        loss_criterion (torch.nn.modules.loss): criterion to use to compute
            the loss
        num_epochs (int): number of epoch to train the model
        device (torch.device): device where the model is stored
        tracker: W&B run (or no-op) that receives per-epoch metrics

    Returns:
        torch.nn.Module: the trained model
    """
    # training process
    set_train_mode(model)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (data, y) in enumerate(train_loader):
            optimizer.zero_grad() 
            # input
            data = data.to(device)
            y = y.to(device)
            
            # training step
            with torch.cuda.amp.autocast():
                outputs  = model(data)
                # Std models in this codebase (MobileNetV2, ResNet, etc.)
                # return (logits, features); CE wants just the logits.
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')

        log_epoch('std', epoch, num_epochs, epoch_loss / len(train_loader), optimizer, tracker)
        scheduler.step() # increase scheduler step for each epoch

    return model

def run_experiment(config:dict,modality:str):
    """Method to run an experiment. Each experiment is composed by n
    runs, defined in the config dictionary, where in each of them a new
    model is trained.

    Args:
        config (dict): Dictionary containing the configuration of the models
            to train.
        modality (str): Model type. One of [std, memory, encoder_memory] where
            std is the standard model, memory is the baseline that uses only the
            memory and encoder_memory is Memory Wrap
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    # get dataset info
    dataset_name = config['dataset_name']
    num_classes = config[dataset_name]['num_classes']


    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()

    # saving/loading stuff
    save = config['save']
    # Differentiate save dirs by which pretraining variant the encoder came
    # from so supcon / simclr / CE-baseline runs don't clobber each other.
    # Infer the tag from the pretrained checkpoint path (e.g. .../supcon/... -> '_supcon').
    suffix = ''
    if FLAGS.pretrained_encoder:
        suffix = '_pretrained'
        for tag in ('supcon', 'simclr', 'hybrid'):
            if f'/{tag}/' in FLAGS.pretrained_encoder:
                suffix = f'_{tag}'
                break
    # Distinguish linear-probe (frozen encoder) from fine-tune runs so they
    # don't overwrite each other in the same models/ directory.
    if FLAGS.freeze_encoder:
        suffix += '_frozen'
    if FLAGS.augment:
        suffix += '_aug'
    if FLAGS.tag:
        suffix += f'_{FLAGS.tag}'
    modality_dir = FLAGS.modality + suffix
    path_saving_model = 'models/{}/{}/{}/{}/'.format(dataset_name,modality_dir, config['model'],config['train_examples'])
    if save and not os.path.isdir(path_saving_model): 
        os.makedirs(path_saving_model)
    
    # optimizer parameters
    learning_rate = float(config['optimizer']['learning_rate'])
    weight_decay = float(config['optimizer']['weight_decay'])
    nesterov = bool(config['optimizer']['nesterov'])
    momentum = float(config['optimizer']['momentum'])
    dict_optim = {'lr' :learning_rate, 'momentum':momentum, 'weight_decay':weight_decay, 'nesterov':nesterov}
    opt_milestones = config[dataset_name]['opt_milestones']

    run_acc = []
    initial_run = 0
    if FLAGS.continue_train:
        # load model
        print("Restarting training process\n")
        info = pickle.load( open(path_saving_model+"conf.p", "rb" ) )
        initial_run = info['run_num']
        run_acc = info['accuracies']
    for run in range(initial_run,config['runs']):
        run_time = time.time()
        utils.set_seed(run)
        model = utils.get_model(config['model'],num_classes,model_type=modality)
        model = model.to(device)
        tracker = tracking.init(
            name=f'{modality_dir}-seed{run}',
            group=f'{dataset_name}-{config["train_examples"]}-{modality_dir}',
            job_type='train',
            config={**config, 'modality': modality, 'seed': run,
                    'pretrained_encoder': FLAGS.pretrained_encoder,
                    'freeze_encoder': FLAGS.freeze_encoder,
                    'augment': FLAGS.augment, 'tag': FLAGS.tag,
                    'val_examples': FLAGS.val_examples})

        # Optional pretrained encoder + optional freeze (linear probe).
        if FLAGS.pretrained_encoder:
            load_pretrained_encoder(model, FLAGS.pretrained_encoder, run, config)
        if FLAGS.freeze_encoder:
            for n, p in model.named_parameters():
                if not n.startswith(HEAD_PREFIXES):
                    p.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError(
                    f"freeze_encoder=True left zero trainable parameters. "
                    f"Model class {type(model).__name__} has no parameter "
                    f"starting with {HEAD_PREFIXES}. Add its head prefix here."
                )

        # training parameters
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],**dict_optim)
        if dataset_name == 'CINIC10':
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config[dataset_name]['num_epochs'])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=opt_milestones)
        # get dataset
        train_loader, val_loader, test_loader, mem_loader = utils.get_loaders(config,run,augment=FLAGS.augment,val_examples=FLAGS.val_examples)
        # Search runs score on held-out training images and never touch test.
        eval_split = 'val' if FLAGS.val_examples else 'test'
        eval_loader = val_loader if FLAGS.val_examples else test_loader

         # training process
        if modality == 'memory' or modality == 'encoder_memory':
            model = train_memory_model(model,[train_loader,mem_loader],optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device=device,tracker=tracker)
            train_time = time.time()

            cum_acc =  []

            # perform 5 times the validation to stabilize results (due to random selection of memory samples)
            init_eval_time = time.time()
            print(f'[eval] Starting 5x evaluation over '
                  f'{len(eval_loader.dataset)} {eval_split} samples '
                  f'(silent until each pass completes)...', flush=True)
            for eval_idx in range(5):
                best_acc, best_loss = utils.eval_memory(model,eval_loader, mem_loader,loss_criterion,device)
                cum_acc.append(best_acc)
                tracker.log({'eval/acc': float(best_acc), 'eval/pass': eval_idx + 1})
                print(f'[eval] {eval_idx+1}/5 acc={best_acc:.2f}  '
                      f'loss={best_loss:.4f}  elapsed={(time.time()-init_eval_time)/60:.1f}min',
                      flush=True)
            best_acc = np.mean(cum_acc)
            end_eval_time = time.time()

        else:
            model = train_std_model(model,train_loader,optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device,tracker=tracker)
            train_time = time.time()
            init_eval_time = time.time()
            best_acc, best_loss  = utils.eval_std(model,eval_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(best_acc)
        tracker.summary[f'{eval_split}/acc'] = float(best_acc)
        tracker.summary[f'{eval_split}/loss'] = float(best_loss)
        tracker.finish()

        # save
        if save and path_saving_model:
            saved_name = "{}.pt".format(run+1)
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save({'model_state_dict':model.state_dict(),
            'train_examples': config['train_examples'],
            'mem_examples':  config[config['dataset_name']]['mem_examples'],
            'model_name': config['model'],
            'num_classes': num_classes, 'modality':modality, 'dataset_name':config['dataset_name'],
            'seed': run, 'pretrained_encoder': FLAGS.pretrained_encoder,
            'val_examples': FLAGS.val_examples} , save_path)
            info = {'run_num':run+1,'accuracies':run_acc}
            pickle.dump( info, open( path_saving_model+"conf.p", "wb" ) )

        # log
        print("Run:{} | Best Loss:{:.4f} | Accuracy {:.2f} | Last Loss: Accuracy:| Mean Accuracy:{:.2f} | Std Dev Accuracy:{:.2f}\tT:{:.2f}min\tE:{:.2f}".format(run+1,best_loss,best_acc, np.mean(run_acc), np.std(run_acc),(train_time -run_time)/60,(end_eval_time -init_eval_time)/60), flush=True)







def main(argv):

    config_file = open(r'config/train.yaml')
    config = apply_overrides(yaml.safe_load(config_file), FLAGS.set)

    print("Model:{}\nSizeTrain:{}\nValExamples:{}\n".format(config['model'], config['train_examples'], FLAGS.val_examples))
    print("Config:", config, flush=True)
    run_experiment(config, FLAGS.modality)

if __name__ == '__main__':
  absl.app.run(main)