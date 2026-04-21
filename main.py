from steps import train_pa, run_dpd
from project import Project

if __name__ == '__main__':
    proj = Project()

    steps = {
        'train_pa': train_pa,
        'run_dpd': run_dpd,
    }

    if proj.step not in steps:
        raise ValueError(f"Step '{proj.step}' not supported.")

    print(f"{'#' * 100}\n# Step: {proj.step.upper():<92}#\n{'#' * 100}")
    steps[proj.step].main(proj)