import models as model
from project import Project
from utils.util import count_net_params


def main(proj: Project):
    """Train PA model."""
    proj.set_device()

    # Build dataloaders
    (train_loader, val_loader, test_loader), input_size = proj.build_dataloaders()

    # Build model
    net = model.CoreModel(
        input_size=input_size,
        hidden_size=proj.PA_hidden_size,
        num_layers=proj.PA_num_layers,
        backbone_type=proj.PA_backbone,
        window_size=proj.window_size,
        num_dvr_units=proj.num_dvr_units
    ).to(proj.device)

    pa_model_id = proj.gen_pa_model_id(count_net_params(net))
    print(f"::: Number of PA Model Parameters: {count_net_params(net)}")

    # Setup training components
    proj.build_logger(model_id=pa_model_id)
    criterion = proj.build_criterion()
    optimizer, lr_scheduler = proj.build_optimizer(net=net)

    # Train
    proj.train(
        net=net, criterion=criterion, optimizer=optimizer,
        lr_scheduler=lr_scheduler, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader,
        best_model_metric='NMSE'
    )