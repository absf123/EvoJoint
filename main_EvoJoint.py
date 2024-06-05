import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # allocate available gpu number (.cuda())
import numpy as np
import pandas as pd
from datetime import datetime
from main_HPO import pbt_hpo
from main_NAO import pbt_nao
from Config import Config


def main_two_stage(subject, model, args):
    config_dict = {}
    for i in range(1, args.num_models+1):
        config_dict.update({f'ens{i}_F1':8, f'ens{i}_T1':125, f'ens{i}_T2':30})  # initial architectures

    iteration_best_hpo_config = {}
    iteration_test_hpo_acc = {}
    iteration_best_nao_config = {}
    iteration_test_nao_acc = {}

    for iter in range(args.num_iteration):
        iteration_best_hpo_config[f'{iter}'] = {}
        iteration_best_nao_config[f'{iter}'] = {}

        print("Iteration: ", iter)
        # Stage1: Hyperpararmeter optimization stage (HPO)
        test_acc_hpo, best_config_hpo = pbt_hpo(subject, model, args, config_dict)
        for i in range(1, args.num_models+1):
            config_dict[f'ens{i}_lr'] = best_config_hpo[f'ens{i}_lr']
            config_dict[f'ens{i}_wd'] = best_config_hpo[f'ens{i}_wd']

        # Stage2: Neural architecture optimization stage (NAO)
        test_acc_nao, best_config_nao = pbt_nao(subject, model, args, config_dict)

        for i in range(1, args.num_models + 1):
            config_dict[f'ens{i}_F1'] = best_config_nao[f'ens{i}_F1']
            config_dict[f'ens{i}_T1'] = best_config_nao[f'ens{i}_T1']
            config_dict[f'ens{i}_T2'] = best_config_nao[f'ens{i}_T2']
            iteration_best_hpo_config[f'{iter}'].update({f'ens{i}_lr': best_config_hpo[f'ens{i}_lr'], f'ens{i}_wd': best_config_hpo[f'ens{i}_wd']})
            iteration_best_nao_config[f'{iter}'].update({f'ens{i}_F1': best_config_nao[f'ens{i}_F1'], f'ens{i}_T1': best_config_nao[f'ens{i}_T1'], f'ens{i}_T2': best_config_nao[f'ens{i}_T2']})
        iteration_best_nao_config[f'{iter}'].update({"subject": best_config_nao['subject'], "model": best_config_nao['model'], "optim": best_config_nao['optim']})

        iteration_test_hpo_acc[f'{iter}'] = test_acc_hpo
        iteration_test_nao_acc[f'{iter}'] = test_acc_nao
        print(f"Iteration {iter + 1} HPO ACC: ", test_acc_hpo)
        print(f"Iteration {iter + 1} NAO ACC: ", test_acc_nao)

    return iteration_test_hpo_acc, iteration_best_hpo_config, \
           iteration_test_nao_acc, iteration_best_nao_config


def cross_subject(model, args):
    TEST_HPO_result = []
    TEST_NAO_result = []

    for subject in range(1, 10):
        print("BCI MI training start...")
        print("""========================START[!] subject id [{}/9] ...========================""".format(subject))

        iteration_test_hpo_acc, _, \
        iteration_test_nao_acc, _ = main_two_stage(subject, model, args)

        TEST_HPO_result.append(iteration_test_hpo_acc)
        TEST_NAO_result.append(iteration_test_nao_acc)

        if subject != 10:
            print("finish subject{}".format(subject))
            print("===== NEXT SUBJECT =====")
        print()

    print("=" * 58)
    print("=" * 15, " AVERAGE ALL SUBJECT TEST ", "=" * 15)
    print("=" * 58)

    result_df = {}
    task = f"BCI subject dependent"
    remark = f"task: {task} model: {model}"

    All_subject_HPO_acc = {str(i): [] for i in range(args.num_iteration)}
    All_subject_NAO_acc = {str(i): [] for i in range(args.num_iteration)}

    for f in range(0, 9):
        for i in range(args.num_iteration):
            All_subject_HPO_acc[f'{i}'].append(TEST_HPO_result[f][f'{i}'])
            All_subject_NAO_acc[f'{i}'].append(TEST_NAO_result[f][f'{i}'])

    for i in range(args.num_iteration):
        result_df[f'TEST HPO ACC iter{i}'] = np.mean(All_subject_HPO_acc[f'{i}'])
        result_df[f'TEST HPO STD iter{i}'] = np.std(All_subject_HPO_acc[f'{i}'], ddof=1)
        result_df[f'TEST NAO ACC iter{i}'] = np.mean(All_subject_NAO_acc[f'{i}'])
        result_df[f'TEST NAO STD iter{i}'] = np.std(All_subject_NAO_acc[f'{i}'], ddof=1)

    result_df['remark'] = [str(remark)]
    result_df = pd.DataFrame(result_df, index=[0])

    print("="*15," TEST ","="*15)
    print("model: ", model)
    for i in range(args.num_iteration):
        print(f"TEST HPO ACC iter{i}: ", result_df[f'TEST HPO ACC iter{i}'] .values)
        print(f"TEST HPO STD iter{i}: ", result_df[f'TEST HPO STD iter{i}'] .values)
        print(f"TEST NAO ACC iter{i}: ", result_df[f'TEST NAO ACC iter{i}'] .values)
        print(f"TEST NAO ACC iter{i}: ", result_df[f'TEST NAO STD iter{i}'] .values)

    return All_subject_HPO_acc, All_subject_NAO_acc


if __name__ == '__main__':
    args = Config()
    model = "EEGNet"
    model_save_timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    print("model_save_timestampe: ", model_save_timestamp)

    All_subject_HPO_acc, All_subject_NAO_acc = cross_subject(model, args)

    print("===============finish training===============")
    print(f"timestamp: {model_save_timestamp}")
    print(f"[!!] Successfully complete..")