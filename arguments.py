import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # Dataset & Log
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--filename', default='')
    parser.add_argument('--log_precision', default=8, type=int)

    # Training
    parser.add_argument('--step', default='run_dpd')
    parser.add_argument('--eval_val', default=1, type=int)
    parser.add_argument('--eval_test', default=1, type=int)
    parser.add_argument('--accelerator', default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--devices', default=4, type=int)
    parser.add_argument('--re_level', default='soft', choices=['soft', 'hard'])
    parser.add_argument('--use_segments', action='store_true')

    # Data processing
    parser.add_argument('--frame_length', default=200, type=int)
    parser.add_argument('--frame_stride', default=1, type=int)

    # Hyperparameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--loss_type', default='l2', choices=['l1', 'l2'])
    parser.add_argument('--opt_type', default='adamw', choices=['sgd', 'adam', 'adamw', 'adabound', 'rmsprop'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--lr_schedule', default=0, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_end', default=1e-4, type=float)
    parser.add_argument('--decay_factor', default=0.1, type=float)
    parser.add_argument('--patience', default=10, type=float)
    parser.add_argument('--grad_clip_val', default=200, type=float)

    # GMP
    parser.add_argument('--K', default=4, type=int)

    # PA model
    backbones = ['gmp', 'gru', 'dgru', 'lstm', 'pgjanet', 'dvrjanet', 'bojanet', 'pnjanet', 'rvtdsmgu', 'smgu']
    parser.add_argument('--PA_backbone', default='dgru', choices=backbones)
    parser.add_argument('--PA_hidden_size', default=8, type=int)
    parser.add_argument('--PA_num_layers', default=1, type=int)

    # DPD model
    parser.add_argument('--DPD_backbone', default='rvtdsmgu', choices=backbones)
    parser.add_argument('--DPD_hidden_size', default=8, type=int)
    parser.add_argument('--DPD_num_layers', default=1, type=int)

    # Quantization
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--n_bits_w', default=8, type=int)
    parser.add_argument('--n_bits_a', default=8, type=int)
    parser.add_argument('--pretrained_model', default='')
    parser.add_argument('--quant_dir_label', default='')
    parser.add_argument('--q_pretrain', default=False, type=bool)

    # Thresholds
    parser.add_argument('--thx', type=float, default=0.0)
    parser.add_argument('--thh', type=float, default=0.0)

    # Model-specific
    parser.add_argument('--num_dvr_units', default=4, type=int)
    parser.add_argument('--window_size', default=4, type=int)

    return parser.parse_args()