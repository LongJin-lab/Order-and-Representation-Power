
import os
import pandas as pd
import numpy as np


def _get_acc(file):
    '''从文件获取acc'''
    f = open(data_path+file, 'r')
    line = f.readlines()[-4:]
    test_acc = line[0].strip('\n').split('Accuracy: ')[1]
    train_acc = line[1].strip('\n').split('Accuracy: ')[1]
    test_noise_acc = line[2].strip('\n').split('Accuracy_pertur: ')[1]
    train_noise_acc = line[3].strip('\n').split('Accuracy_pertur: ')[1]
    return float(test_acc), float(train_acc), float(test_noise_acc), float(train_noise_acc)


def get_coefficient(file_key):
    split = file_key.split('_')
    model = split[0]
    a0 = split[2]
    a1 = split[3]
    a2 = split[4]
    b0 = split[5]
    noise = split[-1]
    return model, float(a0), float(a1), float(a2), float(b0), float(noise)


def _get_train_test_detail(model, a0, a1, a2, b0, noise, root, stab, files):
    '''获取一组参数下所有模型的acc，求均值、标准差'''

    test_accs = []
    train_accs = []
    test_noise_accs = []
    train_noise_accs = []

    for file in files:
        test_acc, train_acc, test_noise_acc, train_noise_acc = _get_acc(file)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        test_noise_accs.append(test_noise_acc)
        train_noise_accs.append(train_noise_acc)

    # 计算均值、标准差
    test_mean, test_std = np.mean(test_accs), np.std(test_accs, ddof=1)
    train_mean, train_std = np.mean(train_accs), np.std(train_accs, ddof=1)
    test_noise_mean, test_noise_std = np.mean(
        test_noise_accs), np.std(test_noise_accs, ddof=1)
    train_noise_mean, train_noise_std = np.mean(
        train_noise_accs), np.std(train_noise_accs, ddof=1)

    return [model, a0, a1, a2, b0, noise, root, stab,
            round(test_mean, 2), round(test_std, 2),
            round(test_noise_mean, 2),
            round(test_noise_std, 2),
            round(train_mean, 2), round(train_std, 2),
            round(train_noise_mean, 2), round(train_noise_std, 2)]


def file2dic(data_path):
    '''将文件按照参数进行分组'''
    file_dic = {}
    for file_name in os.listdir(data_path):
        if '.py' in file_name or '.csv' in file_name:
            continue
        file_key = file_name.split('_tag', 1)[0]
        if not file_dic.get(file_key):
            file_dic[file_key] = [file_name]
        else:
            file_dic.get(file_key).append(file_name)
    return file_dic


def get_root_stab(a0, a1, a2, b0):
    '''从已有表格，根据系数获取root和stab'''
    file_path = '/media3/clm/HighOrderNetStructure/robust_dychf/known.xlsx'
    df = pd.DataFrame(pd.read_excel(file_path, sheet_name='c100'))
    for index, row in df.iterrows():
        if a0 == float(row['a_0']) and a1 == float(row['a_1']) and a2 == float(row['a_2']) and b0 == float(row['b_0']):
            stability = row['Stability']
            root1 = round(float(row['root1']), 2)
            root2 = round(float(row['root2']), 2)
            root3 = round(float(row['root3']), 2)
            return str(root1)+', '+str(root2)+', '+str(root3), stability

    return None, None


if __name__ == '__main__':

    isTest = False

    noise_type = ['randn', 'rand', 'const']
    if isTest:
        data_root = "/media3/clm/HighOrderNetStructure/robust_dychf/test/"
    else:
        data_root = "/media3/clm/HighOrderNetStructure/robust_dychf/allmodel/"
    for type in noise_type:
        data_path = data_root + 'results_'+type+'/'
        file_dic = file2dic(data_path)
        data_arr = []
        for file_key in file_dic.keys():

            # 含零的进行过滤
            model, a0, a1, a2, b0, noise = get_coefficient(file_key)
            if a0 == 0 or a1 == 0 or a2 == 0 or b0 == 0:
                continue

            root, stab = get_root_stab(a0, a1, a2, b0)
            if root is None:
                continue

            files = file_dic.get(file_key)
            tmp_arr = _get_train_test_detail(
                model, a0, a1, a2, b0, noise, root, stab, files)
            data_arr.append(tmp_arr)

        df = pd.DataFrame(data_arr, columns=['Model', 'a0', 'a1', 'a2', 'b0', 'noise',
                                             'Moduli of roots', 'Stab.',
                                             'Test mean', 'Test STD',
                                             'Test noise mean', 'Test noise STD',
                                             'Train mean', 'Train STD',
                                             'Train noise mean', 'Train noise STD'])

        df = df.sort_values(by=['a0', 'a1', 'a2', 'b0', 'noise'])

        print(df)
        save_path = data_root+'csv/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df.to_csv(save_path+type+'.csv')
