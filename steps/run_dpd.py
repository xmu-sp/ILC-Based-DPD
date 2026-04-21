import os
import sys
import pandas as pd
import torch
import models as model
from modules.paths import create_folder
from project import Project
from utils.util import count_net_params
from modules.data_collector import load_dataset

sys.path.append('../..')
from quant import get_quant_model


def main(proj: Project):
    """Run DPD inference and save predistorted signals."""
    proj.set_device()

    # Load test data
    _, _, _, _, X_test, _ = load_dataset(dataset_name=proj.dataset_name)

    # Load PA model info (for path construction)
    net_pa = model.CoreModel(
        input_size=2, hidden_size=proj.PA_hidden_size,
        num_layers=proj.PA_num_layers, backbone_type=proj.PA_backbone,
        num_dvr_units=proj.num_dvr_units
    )
    pa_model_id = proj.gen_pa_model_id(count_net_params(net_pa))

    # Build DPD model
    net_dpd = model.CoreModel(
        input_size=2, hidden_size=proj.DPD_hidden_size,
        num_layers=proj.DPD_num_layers, backbone_type=proj.DPD_backbone
    )
    net_dpd = get_quant_model(proj, net_dpd)

    # Load pretrained weights
    quant_dir = proj.args.quant_dir_label if proj.args.quant else ''
    model_dir = pa_model_id.split('_P_')[0]
    path = os.path.join('save', proj.dataset_name, 'train_dpd', model_dir, quant_dir,
                        proj.gen_dpd_model_id(count_net_params(net_dpd)) + '.pt')

    print(f"::: Loading DPD Model: {path}")
    net_dpd.load_state_dict(torch.load(path))
    net_dpd = net_dpd.to(proj.device).eval()

    print(f"::: Number of DPD Parameters: {count_net_params(net_dpd)}")

    # Run inference
    with torch.no_grad():
        dpd_in = torch.Tensor(X_test).unsqueeze(0).to(proj.device)
        dpd_out = net_dpd(dpd_in).squeeze().cpu()

    # Save results
    out_dir = os.path.join('dpd_out', quant_dir) if quant_dir else 'dpd_out'
    create_folder([out_dir])

    df = pd.DataFrame({'I': X_test[:, 0], 'Q': X_test[:, 1],
                       'I_dpd': dpd_out[:, 0], 'Q_dpd': dpd_out[:, 1]})
    df.to_csv(os.path.join(out_dir, f"{proj.gen_dpd_model_id(count_net_params(net_dpd))}.csv"), index=False)
    print("DPD outputs saved to ./dpd_out")