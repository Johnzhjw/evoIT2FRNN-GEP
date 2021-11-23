import os
import json
import numpy as np


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix): # 判断一个字符的结尾是否是某字符   Python 内置函数 endswith（）
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix): # 判断一个字符的开始是否是某字符   Python 内置函数 endswith（）
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


def read_from_folders(dir="./"):
    scan = ScanFile(dir, postfix="")
    subdirs = scan.scan_subdir()

    print("扫描的文件是:")
    all_results = []
    for subdir in subdirs:
        tar_fnm = subdir.split('-')
        if len(tar_fnm) != 4:
            continue
        else:
            tar_fnm = tar_fnm[1]
        stats = json.load(open(os.path.join(subdir, tar_fnm)))
        stats['alg'] = tar_fnm
        all_results.append(stats)
        print(os.path.join(subdir, tar_fnm))

    probs = {}
    algs  = {}
    mat_res = np.zeros((100, 100))
    for res in all_results:
        prob = '.'.join(res['file_name'].split('/')[-1].split('.')[:-1])
        prb = prob.replace('Stk.','').replace('.all','').replace('.T15','')
        alg = res['alg']
        if res['n_ft'] > 1:
            alg += '-m'
        if prb not in probs:
            ind_prb = len(probs)
            probs[prb] = ind_prb
        else:
            ind_prb = probs[prb]
        if alg not in algs:
            ind_alg = len(algs)
            algs[alg] = ind_alg
        else:
            ind_alg = algs[alg]
        mat_res[ind_prb,ind_alg] = np.mean(res['test_err'])
    mat_res = mat_res[:len(probs),:len(algs)]
    print(mat_res)


def read_from_logs():
    log1 = 'out_run1_all.log'
    log2 = 'out_run1_TS.log'
    all_res = []
    with open(log1, 'r', encoding='utf-8') as f:
        for line in f:
            if 'tensor(' in line:
                val = float(line.split('(')[-1].split(')')[0])
                all_res.append(val)
    with open(log2, 'r', encoding='utf-8') as f:
        for line in f:
            if 'tensor(' in line:
                val = float(line.split('(')[-1].split(')')[0])
                all_res.append(val)
    print(all_res)
    print(len(all_res))
    tmp = np.array(all_res).reshape(-1, 10)
    t_res = []
    for t in tmp:
        t_res.append(t.mean())
    print(t_res)
    t_arr = np.array(t_res).reshape(-1, 7)
    print(t_arr)


if __name__ == "__main__":
    read_from_folders()
    # read_from_logs()
