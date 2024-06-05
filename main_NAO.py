import os
import torch
import torch.nn as nn
from ray import air, tune
from models.EEGNet_model import EEGNet_PBT
from data_loader.data_generator import DataGenerator
from utils.utils import AttrDict
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining


def get_data_loader(subject, batch_size, data_path):
    data_config = AttrDict({
        "subject": subject,
        'folder': data_path,
        "transform":
            {"ToTensor": True},
        'batch_size': batch_size
    })
    data = DataGenerator(data_config)
    return data.train_loader, data.val_loader, data.test_loader


def build_models(num_models, model, config):
    train_models = []
    print("Model: ", model)
    for i in range(num_models):
        model_config = {
            "init_method": 'xavier_uni',
            "F1": config.get(f'ens{i+1}_F1', 8),
            "D": 2,
            "F2": 'auto',
            "T1": config.get(f'ens{i+1}_T1', 125),
            "T2": config.get(f'ens{i+1}_T2', 33),
            "P1": 8,
            "P2": 16,
            "stride": 1,
            "drop_out": 0.5,
            "pool_mode": 'mean'}
        train_model = EEGNet_PBT(input_shape=[None, 1, 22, 1125], n_classes=4, **model_config)

        train_models.append(train_model)
    return train_models


def build_optimizer(model_list, config):
    optim_list = []
    for i in range(len(model_list)):
        optimizer = torch.optim.RMSprop(model_list[i].parameters(),
                                        lr=config.get(f"ens{i+1}_lr", 0.01),
                                        weight_decay=config.get(f"ens{i+1}_wd", 0.01),
                                        momentum=config.get(f"momentum", 0.9))
        optim_list.append(optimizer)
    return optim_list


def train(model_list, optim_list, criterion, train_loader):
    for i in range(len(model_list)):
        model_list[i].train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        for i in range(len(optim_list)):
            optim_list[i].zero_grad()

        loss = 0
        output_list = []
        for i in range(len(model_list)):
            output = model_list[i](data)
            output_list.append(output)
            loss += criterion(output, target)

        loss.backward()

        for i in range(len(optim_list)):
            optim_list[i].step()


def val(model_list, criterion, val_loader):
    for i in range(len(model_list)):
        model_list[i].eval()

    model_correct = {}
    for i in range(len(model_list)):
        model_correct[f'model_ens{i+1}_correct'] = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()

            loss = 0
            output_list = []
            predicted_list = []
            for i in range(len(model_list)):
                output = model_list[i](data)
                output_list.append(output)
                loss += criterion(output, target).item()

                predicted_list.append(torch.max((output.data), 1)[1])

            total += target.size(0)

            for i in range(len(model_list)):
                model_correct[f'model_ens{i+1}_correct'] += (predicted_list[i] == target).sum().item()
            # soft-voting
            weighted_outputs = 0
            for i in range(len(model_list)):
                weighted_outputs += output_list[i]
            _, voting_predicted = torch.max(weighted_outputs, 1)
            correct += (voting_predicted == target).sum().item()

    return loss / len(val_loader), correct / total


def test(model_list, data_loader):
    for i in range(len(model_list)):
        model_list[i].eval()

    model_correct = {}
    for i in range(len(model_list)):
        model_correct[f'model_ens{i + 1}_correct'] = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output_list = []
            predicted_list = []
            for i in range(len(model_list)):
                output = model_list[i](data)
                output_list.append(output)

                predicted_list.append(torch.max((output.data), 1)[1])

            total += target.size(0)

            for i in range(len(model_list)):
                model_correct[f'model_ens{i + 1}_correct'] += (predicted_list[i] == target).sum().item()

            # soft-voting
            weighted_outputs = 0
            for i in range(len(model_list)):
                weighted_outputs += output_list[i]
            _, voting_predicted = torch.max(weighted_outputs, 1)
            correct += (voting_predicted == target).sum().item()

    return model_correct, correct / total, total



def trainer(config):
    print(config['subject'])
    train_loader, val_loader, _ = get_data_loader(config['subject'], config['batch_size'], config['data_path'])
    # Model
    model_list = build_models(num_models=config['num_models'], model=config["model"], config=config)
    # Set device
    for i in range(len(model_list)):
        model_list[i].cuda()

    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optim_list = build_optimizer(model_list, config)

    # Model load from perturbed architecture config
    step = 0
    current_state_dict = [[] for _ in range(len(model_list))]
    for i in range(len(model_list)):
        current_state_dict[i] = model_list[i].state_dict()

    if session.get_checkpoint():
        if session.get_checkpoint():
            print("Loading from checkpoint.")
            loaded_checkpoint = session.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                checkpoint = torch.load(path)
                for i in range(len(model_list)):
                    for name, param in checkpoint[f'model_ens{i+1}_state_dict'].items():
                        if name in current_state_dict[i]:
                            current_param = current_state_dict[i][name]
                            if param.size() != current_param.size():
                                min_size = tuple(min(s1, s2) for s1, s2 in zip(param.size(), current_param.size()))
                                slices = tuple(slice(0, min_size[dim]) for dim in range(len(min_size)))
                                current_param[slices] = param[slices]
                            else:
                                current_state_dict[i][name] = param

                    model_list[i].load_state_dict(current_state_dict[i], strict=False)
                step = checkpoint["step"]

    while True:
        train(model_list, optim_list, criterion, train_loader)
        loss, acc = val(model_list, criterion, val_loader)
        checkpoint = None
        if step % 5 == 0:
            model_dict = {'step': step}
            for i in range(len(model_list)):
                model_dict[f'model_ens{i+1}_state_dict'] = model_list[i].state_dict()

            os.makedirs("my_model", exist_ok=True)
            torch.save(
                model_dict,
                "my_model/checkpoint.pt",
            )
            checkpoint = Checkpoint.from_directory("my_model")

        step += 1
        session.report({"mean_accuracy": acc}, checkpoint=checkpoint)


def pbt_nao(subject, model, args, config_dict):
    print("Neural architecture search")
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    F1_ch_dict = {}
    T1_kernel_dict = {}
    T2_kernel_dict = {}
    F1_ch_space_dict = {}
    T1_kernel_space_dict = {}
    T2_kernel_space_dict = {}
    # Neural architecture search space
    for i in range(1, args.num_models+1):
        F1_ch_dict.update({f'ens{i}_F1': tune.choice([8, 16, 32, 64])})
        T1_kernel_dict.update({f'ens{i}_T1': tune.choice([50, 100, 125, 150, 200, 250])})
        T2_kernel_dict.update({f'ens{i}_T2': tune.choice([10, 20, 30, 40, 50])})
        F1_ch_space_dict.update({f'ens{i}_F1': tune.choice([8, 16, 32, 64])})
        T1_kernel_space_dict.update({f'ens{i}_T1': tune.choice([50, 100, 125, 150, 200, 250])})
        T2_kernel_space_dict.update({f'ens{i}_T2': tune.choice([10, 20, 30, 40, 50])})

    other_key_value = {"subject": subject,
                       "checkpoint_interval": args.perturbation_interval,
                       "num_models":args.num_models,
                       "model": model,
                       "optim": args.optim,
                       "momentum": args.momentum,
                       "folder": args.data_path,
                       "batch_size": args.batch_size,
                       "data_path": args.data_path
                       }

    # fixed hyperparameter
    for i in range(1, args.num_models+1):
        other_key_value.update({f'ens{i}_lr':config_dict[f'ens{i}_lr']})
        other_key_value.update({f'ens{i}_wd':config_dict[f'ens{i}_wd']})

    ## Set scheduler
    # parameter initial
    param_space_dict = {}
    param_space_dict.update(other_key_value)
    # tuning architecture config
    param_space_dict.update(F1_ch_space_dict)
    param_space_dict.update(T1_kernel_space_dict)
    param_space_dict.update(T2_kernel_space_dict)
    # PBT: hyperparam_mutations
    hyperparam_dict = {}
    hyperparam_dict.update(F1_ch_dict)
    hyperparam_dict.update(T1_kernel_dict)
    hyperparam_dict.update(T2_kernel_dict)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=args.perturbation_interval,
        hyperparam_mutations=hyperparam_dict)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainer),
            resources={"cpu": 3, "gpu": 1}  # gpus_per_trial
        ),
        run_config=air.RunConfig(
            name="pbt_test",
            local_dir=args.ray_model_path,
            stop={
                "mean_accuracy": 0.99,
                "training_iteration": args.training_epochs,
            },
            failure_config=air.FailureConfig(
                fail_fast=False,
            ),
        ),

        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric="mean_accuracy",
            mode="max",
            num_samples=args.num_samples,
        ),
        param_space=param_space_dict)

    results = tuner.fit()

    with results.get_best_result().checkpoint.as_directory() as best_checkpoint_path:
        print("result get best result: ", results.get_best_result().config)
        best_trained_model_list = build_models(num_models=args.num_models, model=model, config=results.get_best_result().config)
        device = torch.device('cpu')
        best_checkpoint = torch.load(
            os.path.join(best_checkpoint_path, "checkpoint.pt"), map_location=device)

        for i in range(len(best_trained_model_list)):
            best_trained_model_list[i].load_state_dict(best_checkpoint[f"model_ens{i + 1}_state_dict"])
    _, _, test_loader = get_data_loader(subject, args.batch_size, args.data_path)
    model_test_acc_dict, test_acc, total = test(best_trained_model_list, test_loader)

    print("Best acc: ", test_acc)
    branch_acc = []
    for key, value in model_test_acc_dict.items():
        print(f"{key} acc: {value / total}")
        branch_acc.append(value / total)

    return test_acc, results.get_best_result().config
