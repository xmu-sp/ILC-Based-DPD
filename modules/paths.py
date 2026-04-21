import argparse
import os


def gen_log_stat(args: argparse.Namespace, elapsed_time, net, optimizer, epoch,
                 train_stat=None, val_stat=None, test_stat=None):
    """Generate log statistics dictionary."""
    # Get learning rate
    lr_curr = optimizer.param_groups[0]['lr'] if optimizer else 0

    # Count parameters
    n_param = sum(p.numel() for p in net.parameters())

    # Determine backbone and hidden size based on step
    if args.step == 'train_pa':
        backbone, hidden_size = args.PA_backbone, args.PA_hidden_size
    else:
        backbone, hidden_size = args.DPD_backbone, args.DPD_hidden_size

    log_stat = {
        'EPOCH': epoch, 'N_EPOCH': args.n_epochs, 'TIME:': elapsed_time,
        'LR': lr_curr, 'BATCH_SIZE': args.batch_size, 'N_PARAM': n_param,
        'FRAME_LENGTH': args.frame_length, 'BACKBONE': backbone, 'HIDDEN_SIZE': hidden_size,
    }

    # Add DPD-specific metrics
    if args.step == 'train_dpd' and hasattr(net.dpd_model.backbone, 'thx'):
        log_stat.update({'THX': net.dpd_model.backbone.thx, 'THH': net.dpd_model.backbone.thh})
        if hasattr(net.dpd_model.backbone, 'get_temporal_sparsity'):
            log_stat.update(net.dpd_model.backbone.get_temporal_sparsity())
            net.dpd_model.backbone.set_debug(1)

    # Merge stats with prefixes
    for prefix, stat in [('TRAIN', train_stat), ('VAL', val_stat), ('TEST', test_stat)]:
        if stat:
            log_stat.update({f'{prefix}_{k.upper()}': v for k, v in stat.items()})

    return log_stat


def gen_pa_model_id(args):
    """Generate PA model identifier string."""
    parts = [
        'PA', 'S', str(args.seed), 'M', args.PA_backbone.upper(),
        'H', str(args.PA_hidden_size), 'F', str(args.frame_length)
    ]
    return '_'.join(parts)


def gen_dir_paths(args: argparse.Namespace):
    """Generate directory paths for saving models and logs."""
    base = os.path.join('./save' if 'save' in str(args.step) else './log',
                        args.dataset_name, args.step, args.quant_dir_label)

    if args.step == 'train_pa':
        return (f'./save/{args.dataset_name}/{args.step}/{args.quant_dir_label}',
                f'./log/{args.dataset_name}/{args.step}/{args.quant_dir_label}/history',
                f'./log/{args.dataset_name}/{args.step}/{args.quant_dir_label}/best')
    else:
        pa_id = gen_pa_model_id(args)
        return (f'./save/{args.dataset_name}/{args.step}/{pa_id}/{args.quant_dir_label}',
                f'./log/{args.dataset_name}/{args.step}/{pa_id}/{args.quant_dir_label}/history',
                f'./log/{args.dataset_name}/{args.step}/{pa_id}/{args.quant_dir_label}/best')


def gen_file_paths(save_dir: str, hist_dir: str, best_dir: str, model_id: str):
    """Generate file paths for model and logs."""
    return (f'{save_dir}/{model_id}.pt',
            f'{hist_dir}/{model_id}.csv',
            f'{best_dir}/{model_id}.csv')


def create_folder(folders):
    """Create directories if they don't exist."""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)