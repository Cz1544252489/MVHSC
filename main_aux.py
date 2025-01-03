# Data importation and iteration
import argparse
import json
import os
import time
import zipfile
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity


class data_importation:
    def __init__(self, S):
        self.device = S["device"]
        self.dataset_download_or_not = S["download_or_not"]
        self.dataset_name = S["datasets_name"]
        self.root_path = "./"
        self.dataset_link = "http://mlg.ucd.ie/files/datasets/3sources.zip"
        self.view_num = None
        self.cluster_num = None
        self.sources = None
        self.file_types = None
        self.label_mapping = None
        self.loop_mapping = None
        self.view = None
        self.pre_data_definition(S)
        np.random.seed()
        if self.dataset_download_or_not:
            self.download_dataset()
        self.splited_data = self.split_data()
        self.Theta, self.x, self.y = self.inital()

    def pre_data_definition(self, S):
        match self.dataset_name:
            case "3sources":
                self.sources = ['bbc', 'guardian', 'reuters']
                self.view_num = S["view_num"]
                self.view = S["view"]
                self.file_types = ['mtx', 'terms', 'docs']
                self.cluster_num = 6
                self.label_mapping = {
                    "business": 1,
                    "entertainment": 2,
                    "health": 3,
                    "politics": 4,
                    "sport": 5,
                    "tech": 0
                }
                self.loop_mapping = [
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 2, 0],
                    [0, 2, 2],
                    [1, 2, 1],
                    [1, 2, 2]
                ]


    def download_dataset(self):
        os.makedirs("./logs", exist_ok=True)
        os.makedirs(self.root_path, exist_ok=True)
        zip_path = os.path.join(self.root_path, "temp.zip")

        try:
            response = requests.get(self.dataset_link, stream=True)
            response.raise_for_status()

            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("download has finished")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_path)
            print("unzip has finished")

        except Exception as e:
            print(f"wrong as {e}")

        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    @staticmethod
    def load_csr_matrix(file_path):
        sparse_matrix_coo = mmread(file_path)
        return csr_matrix(sparse_matrix_coo)

    @staticmethod
    def load_data_to_dataframe(filename) -> pd.DataFrame:
        data_list = []
        with open(filename, 'r') as file:
            for line in file:
                category, ids = line.strip().split(': ')
                for id in ids.split(','):
                    data_list.append({'category': category, 'id': int(id)})

        df = pd.DataFrame(data_list)
        return df

    def split_data(self) -> dict:
        sources = self.sources
        file_types = self.file_types
        dataset_path = os.path.join(self.root_path, self.dataset_name)

        splited_data = {}
        for source in sources:
            for ftype in file_types:
                filename = f"3sources_{source}.{ftype}"
                filepath = os.path.join(dataset_path, filename)

                match ftype:
                    case 'mtx':
                        splited_data[f"{source}_{ftype}"] = self.load_csr_matrix(filepath)
                    case 'docs'|'terms':
                        tmp = pd.read_csv(filepath, header=None, names=[ftype.capitalize()])
                        splited_data[f"{source}_{ftype}"] = tmp.values.reshape([-1])

        disjoint_clist_filename = "3sources.disjoint.clist"
        disjoint_clist_path = os.path.join(dataset_path, disjoint_clist_filename)
        overlap_clist_filename = "3sources.overlap.clist"
        overlap_clist_path = os.path.join(dataset_path, overlap_clist_filename)

        splited_data['disjoint_clist'] = self.load_data_to_dataframe(disjoint_clist_path).values
        splited_data['overlap_clist'] = self.load_data_to_dataframe(overlap_clist_path).values
        return splited_data

    def get_labels_true(self, view:Literal["bbc","guardian","reuters"]="bbc",
                                   clist_type:Literal["disjoint","overlap"]="disjoint"):
        data = self.splited_data
        m = self.label_mapping
        docs = data[f"{view}_docs"]
        clist = data[f"{clist_type}_clist"]
        labels_true = np.zeros(len(docs),dtype=int)
        for i in range(len(docs)):
            label = clist[np.argwhere(clist==docs[i])[0][0]][0]
            labels_true[i] = m[label]

        return labels_true

    def get_labels_from_sample(self, sample, clist_type:Literal["disjoint","overlap"]="disjoint"):
        data = self.splited_data
        m = self.label_mapping
        clist = data[f"{clist_type}_clist"]
        labels_ture = np.zeros(len(sample), dtype=int)
        for i in range(len(sample)):
            label = clist[np.argwhere(clist==sample[i])[0][0]][0]
            labels_ture[i] = m[label]
        return labels_ture

    def process_data(self) -> dict:
        data0 = self.splited_data
        sources = self.sources
        loop = self.loop_mapping
        data = {"view_num": self.view_num, "sources": sources, "cluster_num": self.cluster_num}
        match self.view_num:
            case 2:
                for i,j,k in loop[self.view:self.view+2]:
                    data[f"docs_{sources[i]}_{sources[j]}"] = np.intersect1d(data0[f"{sources[i]}_docs"],
                                                                             data0[f"{sources[j]}_docs"])

                    data[f"labels_true_{sources[i]}_{sources[j]}"] = self.get_labels_from_sample(
                        data[f"docs_{sources[i]}_{sources[j]}"])
                    temp = [np.where(data0[f"{sources[k]}_docs"] == value)[0].item() for value in
                            data[f"docs_{sources[i]}_{sources[j]}"]]
                    data[f"{sources[k]}_mtx_{sources[i]}_{sources[j]}"] = data0[f"{sources[k]}_mtx"].T[temp]

        return data

    def get_H(self, X):
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            top_p_indices = np.argsort(X[i])[-self.cluster_num:]
            result[i, top_p_indices] = 1

        return result

    def get_Theta_and_F(self, X):
        p = self.cluster_num
        sim = cosine_similarity(X)
        H = self.get_H(sim)
        m, n = H.shape
        W = np.diag(np.random.rand(m))
        D_e = np.zeros(m)
        for i in range(m):
            D_e[i] = sum(H[:, i])
        D_e = np.diag(D_e)
        D_v = np.zeros(n)
        for j in range(n):
            D_v[j] = sum(H[j, :])
        D_v = np.diag(D_v)
        D_v_inv_sqrt = np.linalg.inv(np.sqrt(D_v))
        D_e_inv = np.linalg.inv(D_e)
        Theta = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt
        _, F = eigsh(Theta, p, which='LM')
        Theta = Theta.astype(np.float32)
        F = F.astype(np.float32)
        return Theta, F

    def inital(self):
        data = self.process_data()
        sources = self.sources
        p = self.cluster_num
        map = ["UL", "LL"]
        Theta = {}
        F = {}
        match self.view_num:
            case 2:
                l = 0
                for i,j,k in self.loop_mapping[self.view:self.view+2]:
                    Theta[f"{map[l]}"], F[f"{map[l]}"] = self.get_Theta_and_F(
                        data[f"{sources[k]}_mtx_{sources[i]}_{sources[j]}"])
                    Theta[f"{map[l]}"] = torch.tensor(Theta[f"{map[l]}"])
                    F[f"{map[l]}"] = torch.tensor(F[f"{map[l]}"])
                    l += 1

        return Theta, F["UL"], F["LL"]


class iteration:
    def __init__(self, IN, S):
        self.device = IN.device
        self.x = IN.x
        self.y = IN.y
        self.Theta_y = IN.Theta["LL"]
        self.Theta_x = IN.Theta["UL"]
        self.O = torch.zeros(self.x.shape[0], dtype=torch.float32, device=self.device)
        self.I = torch.eye(self.x.shape[0], dtype=torch.float32, device=self.device)

        self.UL = None
        self.LL = None
        self.lam = None
        self.learning_rate = None
        self.grad = None
        self.loop0 = None
        self.loop1 = None
        self.loop2 = None
        self.proj_x = None
        self.proj_y = None
        self.orth_x = None
        self.orth_y = None
        self.log_filename = None
        self.log_data = {}
        self.alpha = None
        self.beta = None
        self.mu = None
        self.s_u = None
        self.s_l = None
        self.Z = None
        self.epsilon = None
        self.cl = {}
        self.opt_method = None
        self.hypergrad_method = None
        self.pre_definition(S)

    def pre_definition(self, S):
        self.lam = S["lam"]
        self.learning_rate = S["lr"]
        self.UL = self.upper_level(self.lam)
        self.LL = self.lower_level(self.lam, self.Theta_y)
        self.proj_x = S["proj_x"]
        self.proj_y = S["proj_y"]
        self.orth_x = S["orth_x"]
        self.orth_y = S["orth_y"]
        self.alpha = lambda x:1/(1+x)
        self.beta = lambda x:1/(1+x)
        self.mu = S["mu"]
        self.s_l = S["s_l"]
        self.s_u = S["s_u"]
        self.loop0 = S["loop0"]
        self.loop1 = S["loop1"]
        self.loop2 = S["loop2"]
        self.opt_method = S["opt_method"]
        self.hypergrad_method = S["hypergrad_method"]
        self.epsilon = S["epsilon"]

        self.log_filename = S["log_filename"]
        self.log_data.update(S)
        self.log_data["LL_dval"] = []
        self.log_data["UL_dval"] = []
        self.log_data["LL_ngrad_y"] = []
        self.log_data["UL_ngrad_x"] = []
        self.log_data["time_elapsed"] = []

    class lower_level:
        def __init__(self, lam, Theta_y):
            self.lam = lam
            self.Theta_y = Theta_y
            self.val = None
            temp = torch.linalg.svd(Theta_y).S
            self.max_val = torch.topk(temp,6).values.sum() + lam * 6
            self.dval = None
            self.grad = {}
            self.hess = {}

        def cost(self, x, y):
            term1 = torch.trace(y.T @ self.Theta_y @ y)
            term2 = self.lam * torch.trace(y @ y.T @ x @ x.T)
            self.val = term1 + term2
            self.dval = self.max_val-self.val

        def ggrad(self, x, y):
            self.grad["x"] = 2 * self.lam * y @ y.T @ x
            self.grad["y"] = 2 * (self.Theta_y + self.lam * x @ x.T) @ y
            return self.grad

        def ghess(self, x, y):
            self.hess["xx"] = 2 * self.lam * y @ y.T
            self.hess["xy"] = 2 * self.lam * (y @ x.T + x @ y.T)  # 混合二阶偏导
            self.hess["yy"] = 2 * self.Theta_y + 2 * self.lam * x @ x.T
            return self.hess

    class upper_level:
        def __init__(self, lam):
            self.lam = lam
            self.val = None
            self.max_val = 6
            self.dval = None
            self.grad = {}
            self.hess = {}

        def cost(self, x, y):
            self.val = self.lam * torch.trace(y @ y.T @ x @ x.T)
            self.dval = self.max_val-self.val

        def ggrad(self, x, y):
            self.grad["x"] = 2 * self.lam * y @ y.T @ x
            self.grad["y"] = 2 * self.lam * x @ x.T @ y
            return self.grad

        def ghess(self, x, y):
            self.hess["xx"] = 2 * self.lam * y @ y.T
            self.hess["xy"] = 2 * self.lam * (y @ x.T + x @ y.T)  # 混合二阶偏导
            self.hess["yy"] = 2 * self.lam * x @ x.T
            return self.hess

    def update_value(self, x, grad, flag:bool=False):
        x = x + self.learning_rate * grad
        if flag:
            x, _ = torch.linalg.qr(x, mode="reduced")
        return x

    def proj(self, vector, y, flag:bool=False):
        if flag:
            vector = (self.I-y@y.T) @ vector
        return vector

    def run_as_adm(self):
        start_time = time.time()
        for epoch in range(self.loop0):
            for i in range(self.loop1):
                self.LL.cost(self.x, self.y)
                self.LL.ggrad(self.x, self.y)
                self.y = self.update_value(self.y, self.proj(self.LL.grad["y"], self.y, self.proj_y), self.orth_y)
            for j in range(self.loop2):
                self.UL.cost(self.x, self.y)
                self.UL.ggrad(self.x, self.y)
                self.x = self.update_value(self.x, self.proj(self.UL.grad["x"], self.x, self.proj_x), self.orth_x)
            self.log_data["LL_dval"].append(self.LL.dval.item())
            self.log_data["UL_dval"].append(self.UL.dval.item())
            self.log_data["LL_ngrad_y"].append(torch.linalg.norm(self.LL.grad["y"], ord=2).item())
            self.log_data["UL_ngrad_x"].append(torch.linalg.norm(self.UL.grad["x"], ord=2).item())
            self.log_data["time_elapsed"].append(time.time()-start_time)
            if self.UL.dval.item() <= self.epsilon:
                break
        self.log_result()

    def run_as_bda_forward(self):
        start_time = time.time()
        for epoch in range(self.loop0):
            self.Z = self.O
            for i in range(self.loop1):
                self.LL.ggrad(self.x, self.y)
                self.UL.ggrad(self.x, self.y)
                p_u = self.mu * self.alpha(i) * self.s_u
                p_l = (1-self.mu) * self.beta(i) * self.s_l
                grad_y = self.proj(p_u * self.UL.grad["y"] + p_l * self.LL.grad["y"], self.y, self.proj_y)
                self.y = self.update_value(self.y, grad_y, self.orth_y)

                self.LL.ghess(self.x, self.y)
                self.UL.ghess(self.x, self.y)
                A = (p_u + p_l) * self.UL.hess["xy"]
                B = self.I + (p_u * self.UL.hess["yy"] + p_l * self.LL.hess["yy"])
                self.Z = B @ self.Z + A

            grad_x = self.proj(self.lam * self.UL.grad["x"] + self.Z @ self.LL.grad["x"], self.x, self.proj_x)

            for j in range(self.loop2):
                self.x = self.update_value(self.x, grad_x, self.orth_x)

            self.LL.cost(self.x, self.y)
            self.UL.cost(self.x, self.y)
            self.log_data["LL_dval"].append(self.LL.dval.item())
            self.log_data["UL_dval"].append(self.UL.dval.item())
            self.log_data["LL_ngrad_y"].append(torch.linalg.norm(self.LL.grad["y"], ord=2).item())
            self.log_data["UL_ngrad_x"].append(torch.linalg.norm(self.UL.grad["x"], ord=2).item())
            self.log_data["time_elapsed"].append(time.time() - start_time)
            if self.UL.dval.item() <= self.epsilon:
                break
        self.log_result()

    def run_as_bda_backward(self):
        start_time = time.time()
        for epoch in range(self.loop0):
            for i in range(self.loop1):
                self.LL.ggrad(self.x, self.y)
                self.UL.ggrad(self.x, self.y)
                p_u = self.mu * self.alpha(i) * self.s_u
                p_l = (1-self.mu) * self.beta(i) * self.s_l
                grad_y = self.proj(p_u * self.UL.grad["y"] + p_l * self.LL.grad["y"], self.y, self.proj_y)

                self.LL.ghess(self.x, self.y)
                self.UL.ghess(self.x, self.y)
                self.cl[f"{i+1}_B"] = p_u * self.UL.hess["xy"] + p_l * self.UL.hess["xx"]
                self.cl[f"{i+1}_A"] = -self.I + p_l * self.UL.hess["yy"] + p_l * self.UL.hess["xy"]

                self.y = self.update_value(self.y, grad_y, self.orth_y)

            grad_x = 0
            self.UL.ggrad(self.x, self.y)
            self.cl[f"{self.loop1}_alpha"] = self.UL.grad["x"]
            for i in range(self.loop1-1,-1,-1):
                grad_x = grad_x + self.cl[f"{i+1}_B"].T @ self.cl[f"{i+1}_alpha"]
                self.cl[f"{i}_alpha"] = self.cl[f"{i+1}_A"].T @ self.cl[f"{i+1}_alpha"]

            for j in range(self.loop2):
                self.x = self.update_value(self.x, grad_x, self.orth_x)

            self.LL.cost(self.x, self.y)
            self.UL.cost(self.x, self.y)
            self.log_data["LL_dval"].append(self.LL.dval.item())
            self.log_data["UL_dval"].append(self.UL.dval.item())
            self.log_data["LL_ngrad_y"].append(torch.linalg.norm(self.LL.grad["y"], ord=2).item())
            self.log_data["UL_ngrad_x"].append(torch.linalg.norm(self.UL.grad["x"], ord=2).item())
            self.log_data["time_elapsed"].append(time.time() - start_time)
            if self.UL.dval.item() <= self.epsilon:
                break
        self.log_result()

    def run(self):
        match self.opt_method:
            case "ADM":
                self.loop1 = 1
                self.loop2 = 1
                self.run_as_adm()
            case "BDA":
                print(self.hypergrad_method)
                if self.hypergrad_method == "backward":
                    self.run_as_bda_backward()
                elif self.hypergrad_method == "forward":
                    self.run_as_bda_forward()

    def log_result(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{self.log_filename}_{timestamp}.json","w") as file:
            json.dump(self.log_data, file, indent=4)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parser():
    parser = argparse.ArgumentParser(description="None")

    parser.add_argument('--datasets_name', type=str, default="3sources")
    parser.add_argument('--download_or_not', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--view_num', type=int, default=2)
    parser.add_argument('-v','--view', type=int, choices=[0,2,4], default=2)
    parser.add_argument('--seed_num', type=int, default=44)

    parser.add_argument('-o','--opt_method', type=str, choices=["BDA", "RHG", "ADM"],
                        default="BDA")
    parser.add_argument('-m','--hypergrad_method', type=str, choices=["backward", "forward"],
                        default="forward")

    parser.add_argument('-E','--loop0', type=int, default=100)
    parser.add_argument('--loop1', type=int, default=5)
    parser.add_argument('--loop2', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=1e-5)

    parser.add_argument('--s_u', type=float, default=0.5)
    parser.add_argument('--s_l', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--proj_x', type=str2bool, default=True)
    parser.add_argument('--proj_y', type=str2bool, default=True)
    parser.add_argument('--orth_x', type=str2bool, default=True)
    parser.add_argument('--orth_y', type=str2bool, default=True)

    parser.add_argument('--log_filename', type=str, default="./logs/test")

    S0 = parser.parse_args()
    S = vars(S0)
    return S

def create_instance(S:dict):
    DI = data_importation(S)
    IT = iteration(DI, S)

    return IT