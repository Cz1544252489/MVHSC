# This file aims to collect the auxiliary function of MVHSC.
import os
import platform

import argparse
from random import choices
import time
from datetime import datetime
# os.environ["OMP_NUM_THREADS"] = "1"

from scipy.optimize import linear_sum_assignment
from typing import Literal
import pandas as pd
import math
import os
import json
from scipy.sparse.linalg import eigsh
from scipy.io import mmread
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.preprocessing import normalize

# function part

# data importation

class data_importation:
    def __init__(self, S):
        device = torch.device(
            "mps" if S["device_set"] and platform.system() == "Darwin" and torch.backends.mps.is_available()
            else "cuda" if S["device_set"] and torch.cuda.is_available()
            else "cpu"
        )
        self.S = S
        self.view_num = S["view_num"]
        self.view2 = S["view2"]
        self.device = device
        self.cluster_num = 6
        self.root_path = S["root_path"]
        self.sources = ['bbc', 'guardian', 'reuters']
        self.file_types = ['mtx', 'terms', 'docs']
        self.mapping = {
            "business": 1,
            "entertainment": 2,
            "health": 3,
            "politics": 4,
            "sport": 5,
            "tech": 0
        }
        self.mapping1 = [
            [0,1,0],
            [0,1,1],
            [0,2,0],
            [0,2,2],
            [1,2,1],
            [1,2,2]
        ]
        self.data0 = self.get_data0()
        self.data = self.get_data()
        np.random.seed(S["seed_num"])

    @staticmethod
    def normalize_data(X : csr_matrix):
        sparse_matrix_normalized = normalize(X, norm="l2", axis=1)
        return sparse_matrix_normalized

    @staticmethod
    def load_csr_matrix(file_path) -> csr_matrix:
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

        # 创建DataFrame
        df = pd.DataFrame(data_list)
        return df

    def get_data0(self) -> dict:

        # 设置根路径
        root_path = self.root_path

        # 新闻源和文件类型
        sources = self.sources
        file_types = self.file_types

        data0 = {}
        # 使用循环创建变量并加载数据
        for source in sources:
            for ftype in file_types:
                filename = f"3sources_{source}.{ftype}"
                filepath = os.path.join(root_path, filename)

                # 根据文件类型选择读取方法
                if ftype == 'mtx':
                    data0[f"{source}_{ftype}"] = self.load_csr_matrix(filepath)
                else:
                    data0[f"{source}_{ftype}"] = pd.read_csv(filepath, header=None, names=[ftype.capitalize()])

        # 调用函数并打印结果
        disjoint_clist_filename = "3sources.disjoint.clist"
        disjoint_clist_path = os.path.join(root_path, disjoint_clist_filename)
        overlap_clist_filename = "3sources.overlap.clist"
        overlap_clist_path = os.path.join(root_path, overlap_clist_filename)

        data0['disjoint_clist'] = self.load_data_to_dataframe(disjoint_clist_path)
        data0['overlap_clist'] = self.load_data_to_dataframe(overlap_clist_path)

        for key, values in data0.items():
            globals()[key] = values


        return {'bbc_mtx':bbc_mtx, 'bbc_terms':bbc_terms.values.reshape([-1]), 'bbc_docs': bbc_docs.values.reshape([-1]) ,
                'guardian_mtx': guardian_mtx, 'guardian_terms': guardian_terms.values.reshape([-1]), 'guardian_docs': guardian_docs.values.reshape([-1]),
                'reuters_mtx':reuters_mtx, 'reuters_terms':reuters_terms.values.reshape([-1]), 'reuters_docs': reuters_docs.values.reshape([-1]) ,
                'disjoint_clist':disjoint_clist.values, 'overlap_clist':overlap_clist.values}


    def get_labels_true(self,view:Literal["bbc","guardian","reuters"]="bbc",
                                   clist_type:Literal["disjoint","overlap"]="disjoint"):
        data0 = self.data0
        mapping = self.mapping
        docs = data0[f"{view}_docs"]
        clist = data0[f"{clist_type}_clist"]
        labels_true = np.zeros(len(docs),dtype=int)
        for i in range(len(docs)):
            label = clist[np.argwhere(clist==docs[i])[0][0]][0]
            labels_true[i] = mapping[label]

        return labels_true

    def get_labels_from_sample(self, sample, clist_type:Literal["disjoint","overlap"]="disjoint"):
        data0 = self.data0
        mapping = self.mapping
        clist = data0[f"{clist_type}_clist"]
        labels_ture = np.zeros(len(sample), dtype=int)
        for i in range(len(sample)):
            label = clist[np.argwhere(clist==sample[i])[0][0]][0]
            labels_ture[i] = mapping[label]
        return labels_ture

    def get_data(self) -> dict:

        # 按照文件名生成数据
        data0 = self.data0
        sources = self.sources
        data = {"view_num":self.view_num, "sources":sources,
                            "mapping1":self.mapping1, "cluster_num": self.cluster_num}
        match self.view_num:
            case 1:
                for view in sources:
                    data[f"{view}_labels_true"] = self.get_labels_true(view=view)
                    data[f"{view}_mtx"] = data0[f"{view}_mtx"].T
            case 2:
                for i,j,k in self.mapping1[self.view2:self.view2+2]:
                # for i, j, k in [[0,1,0],[0,1,1],[0,2,0],[0,2,2],[1,2,1],[1,2,2]]:
                    data[f"docs_{sources[i]}_{sources[j]}"] = np.intersect1d(data0[f"{sources[i]}_docs"],
                                                                              data0[f"{sources[j]}_docs"])

                    data[f"labels_true_{sources[i]}_{sources[j]}"] = self.get_labels_from_sample(data[f"docs_{sources[i]}_{sources[j]}"])
                    temp = [np.where(data0[f"{sources[k]}_docs"] == value)[0].item() for value in
                            data[f"docs_{sources[i]}_{sources[j]}"]]
                    data[f"{sources[k]}_mtx_{sources[i]}_{sources[j]}"] = data0[f"{sources[k]}_mtx"].T[temp]

            case 3:
                data[f"docs_3sources"] = np.intersect1d(np.intersect1d(data0[f"{sources[0]}_docs"], data0[f"{sources[1]}_docs"])
                                                        , data0[f"{sources[2]}_docs"])
                data[f"labels_true_3sources"] = self.get_labels_from_sample(data[f"docs_3sources"])
                for i in range(3):
                    temp = [np.where(data0[f"{sources[i]}_docs"] == value)[0].item() for value in data["docs_3sources"]]
                    data[f"{sources[i]}_mtx_3sources"] = data0[f"{sources[i]}_mtx"].T[temp]


        return data



class initialization:

    def __init__(self, DI):
        self.mapping2 = ["UL","LL"]
        self.mapping1 = DI.mapping1
        self.view2 = DI.view2
        self.data = DI.data
        self.device = DI.device
        self.S = DI.S
        self.Theta, self.x, self.y = self.initial(self.data["view_num"])

    @staticmethod
    def get_similarity_matrix(X):
        # if method == "euclidean":
        #     dist_matrix = distance.squareform(distance.pdist(X, 'euclidean'))
        #     similarity_matrix = np.exp(-dist_matrix ** 2 / (2. * dist_matrix.std() ** 2))
        # elif method == "cosine":
        similarity_matrix = cosine_similarity(X)
        # else:
        #     raise ValueError("输入错误")
        return similarity_matrix

    def initial(self, view_num, backend="torch"):
        data = self.data
        sources = data["sources"]
        p = data["cluster_num"]
        Theta = {}
        F = {}
        match view_num:
            case 2:
                l = 0
                for i,j,k in self.mapping1[self.view2:self.view2+2]:
                    Theta[f"{self.mapping2[l]}"], F[f"{self.mapping2[l]}"] = self.get_Theta_and_F(data[f"{sources[k]}_mtx_{sources[i]}_{sources[j]}"], p)
                    l += 1

        if backend=="torch":
            Theta = {key: torch.from_numpy(value) for key, value in Theta.items()}
            F = {key: torch.from_numpy(value) for key, value in F.items()}

        x = F["UL"]
        y = F["LL"]
        return Theta, x, y

    @staticmethod
    def get_H(X, p):
        # 创建一个和原矩阵相同形状的全零矩阵
        result = np.zeros_like(X)

        # 对每一行处理
        for i in range(X.shape[0]):
            # 找到最大的p个元素的索引
            top_p_indices = np.argsort(X[i])[-p:]
            # 将这些位置的元素设置为1
            result[i, top_p_indices] = 1

        return result

    def get_Theta_and_F(self, X, p: int):
        sim = self.get_similarity_matrix(X)
        H = self.get_H(sim, p)
        m = H.shape[0]
        n = H.shape[1]
        # W = np.diag(np.ones(m))
        W = np.diag(np.random.rand(m))
        D_e = np.zeros(m)
        for i in range(m):
            D_e[i] = sum(H[:, i])

        D_e = np.diag(D_e)

        D_v = np.zeros(n)
        for i in range(n):
            D_v[i] = sum(H[i, :])

        D_v = np.diag(D_v)

        D_v_inv_sqrt = np.linalg.inv(np.sqrt(D_v))
        D_e_inv = np.linalg.inv(D_e)
        Theta = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt

        _, F = eigsh(Theta, p, which='LM')

        return Theta, F

class evaluation:

    def __init__(self, DI):
        self.data = DI.data
        self.mapping1 = DI.mapping1
        self.view2 = DI.view2
        self.S = DI.S
        mapping = {
            "business": 1,
            "entertainment": 2,
            "health": 3,
            "politics": 4,
            "sport": 5,
            "tech": 0
        }
        self.mapping = mapping

    @staticmethod
    def show_result(X, labels):
        # plt.figure()
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        # cmap的选择还有'rainbow', 'jet'等
        plt.title('PCA of 6D data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter)
        plt.show()

    def show_high_dimension_result(self, F, labels):
        pca = PCA(n_components=2)
        X = pca.fit_transform(F)

        plt.figure(figsize=(8, 6))
        self.show_result(X, labels)

    def cluster(self, F, num_type, method:["normal","spectral"], show=False):
        kmeans = KMeans(n_clusters=num_type, random_state=0)
        spectral = SpectralClustering(n_clusters=num_type, affinity='nearest_neighbors',
                                      assign_labels='kmeans', random_state=42)

        if method == "spectral":
            labels = spectral.fit_predict(F)
        elif method == "normal":
            labels = kmeans.fit_predict(F)
        else:
            raise ValueError("Wrong Input!")
        if show == True:
            self.show_high_dimension_result(F, labels)

        return labels

    def assess(self, data1):
        i,j = self.mapping1[self.view2][0:2]
        sources = self.data["sources"]
        labels_pred = self.cluster(data1, self.data["cluster_num"], self.S["cluster_method"])
        labels_true = self.data[f"labels_true_{sources[i]}_{sources[j]}"]
        nmi = self.calculate_nmi(labels_true, labels_pred)
        ari = self.calculate_ari(labels_true, labels_pred)
        acc = self.calculate_acc(labels_true, labels_pred)
        f1 = self.calculate_acc(labels_true, labels_pred)
        return acc, nmi, ari, f1

    def replace(self, labels):
        replaced_list = [self.mapping[item] for item in labels]
        return replaced_list

    @staticmethod
    def calculate_nmi(labels_true, labels_pred, method:Literal["min","geometric","arithmetic","max"]="min"):
        # 初步看，使用"min" 参数有好的结果
        return normalized_mutual_info_score(labels_true, labels_pred,average_method=method)

    @staticmethod
    def calculate_ari(labels_true, labels_pred):
        return adjusted_rand_score(labels_true, labels_pred)

    def calculate_acc(self, labels_true, labels_pred):
        reordered_labels = self.best_map(labels_true, labels_pred)
        ratio = np.sum(labels_true == reordered_labels)/len(labels_true)
        return ratio

    @staticmethod
    def calculate_f1_score(labels_true, labels_pred):
        f1 = f1_score(labels_true, labels_pred)
        return f1

    @staticmethod
    def judge_orth(F):
        FTF = F.T @ F
        norm2 = torch.linalg.norm(FTF, ord = 2)
        return norm2

    @staticmethod
    def judge_orths(F_UL, F_LL):
        norm_LL = judge_orth(F_LL)
        norm_UL = judge_orth(F_UL)
        print(f"norm_UL:{norm_UL},norm_LL:{norm_LL}")

    @staticmethod
    def record(result, type:Literal["UL", "LL"], **kwargs):
        match type:
            case "UL":
                result["ul_acc"].append(kwargs["acc"])
                result["ul_val"].append(kwargs["val"])
                result["ul_nmi"].append(kwargs["nmi"])
                result["ul_ari"].append(kwargs["ari"])
                result["ul_f1"].append(kwargs["f1"])
                result["norm_grad_ul"].append(kwargs["grad"])

            case "LL":
                result["ll_acc"].append(kwargs["acc"])
                result["ll_val"].append(kwargs["val"])
                result["ll_nmi"].append(kwargs["nmi"])
                result["ll_ari"].append(kwargs["ari"])
                result["ll_f1"].append(kwargs["f1"])
                result["norm_grad_ll"].append(kwargs["grad"])
        return result

    def plot_result(self,data, flag, method:["save","show"],*,picname):
        result = self.output_type(data, flag)
        num_plots = len(result.keys())  # 获取总图数
        cols = 2  # 每行2张图
        rows = (num_plots + cols - 1) // cols  # 根据总图数计算行数

        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))  # 设置子图
        axes = axes.flatten()  # 将 axes 转换为一维数组，便于索引

        for i, key in enumerate(result.keys()):
            ax = axes[i]
            ax.plot(result[f"{key}"], marker='o', linestyle='-')
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.set_ylabel("Value")
            ax.grid(True)

        # 隐藏多余的子图（如果子图多余图数）
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        match method:
            case "show":
                plt.show()
            case "save":
                plt.savefig(picname, format='png', dpi=300)

    @staticmethod
    def use_result(data, method: Literal["dump","load"], file_name):
        match method:
            case "dump":
                with open(file_name, "w") as file:
                    json.dump(data, file, indent=4)
            case "load":
                with open(file_name, "r") as file:
                    data = json.load(file)
                return data

    def result_file_name(self):
        if self.S['plus_datetime']:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"logs/{self.S['file_name']}_{timestamp}.json"
        else:
            return f"logs/{self.S['file_name']}.json"

    def result_fig_name(self):
        if self.S['plus_datetime']:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"logs/{self.S['figure_name']}_{timestamp}.jpg"
        else:
            return f"logs/{self.S['figure_name']}.jpg"

    @staticmethod
    def output_type(result, flag):
        output = {}
        if "val" in flag:
            output["ll_val"] =result["ll_val"]
            output["ul_val"] =result["ul_val"]
        if "grad" in flag:
            output["norm_grad_ll"] =  result["norm_grad_ll"]
            output["norm_grad_ul"] = result["norm_grad_ul"]
        if "nmi" in flag :
            output["ll_nmi"] = result["ll_nmi"]
            output["ul_nmi"] = result["ul_nmi"]
        if "acc" in flag :
            output["ll_acc"] = result["ll_acc"]
            output["ul_acc"] = result["ul_acc"]
        if "ari" in flag :
            output["ll_ari"] = result["ll_ari"]
            output["ul_ari"] = result["ul_ari"]
        if "f1" in flag:
            output["ll_f1"] =result["ll_f1"]
            output["ul_f1"] =result["ul_f1"]

        return output

    @staticmethod
    def best_map(L1, L2):
        """
        Reorder labels in L2 to best match L1 using the Hungarian method.

        Parameters:
        - L1: array-like, ground truth labels (1D array).
        - L2: array-like, labels to be permuted to match L1 (1D array).

        Returns:
        - newL2: array-like, reordered labels for L2.
        """
        L1 = np.asarray(L1).ravel()
        L2 = np.asarray(L2).ravel()

        # Ensure L1 and L2 have the same size
        if L1.shape != L2.shape:
            raise ValueError("L1 and L2 must have the same size.")

        # Get unique labels and their counts in L1 and L2
        unique_L1 = np.unique(L1)
        unique_L2 = np.unique(L2)

        nClass1 = len(unique_L1)
        nClass2 = len(unique_L2)
        nClass = max(nClass1, nClass2)

        # Build the cost matrix
        G = np.zeros((nClass, nClass), dtype=int)
        for i, label1 in enumerate(unique_L1):
            for j, label2 in enumerate(unique_L2):
                G[i, j] = np.sum((L1 == label1) & (L2 == label2))

        # Apply the Hungarian algorithm on the negative cost matrix to maximize the match
        row_ind, col_ind = linear_sum_assignment(-G)

        # Create a new array for L2 with reordered labels
        newL2 = np.zeros_like(L2)
        for i, j in zip(row_ind, col_ind):
            newL2[L2 == unique_L2[j]] = unique_L1[i]

        return newL2

    def process_json_files_multi_keys(self):
        """
        遍历指定目录中以指定前缀开头的 JSON 文件，提取多组键的值。

        :param directory: str - 要搜索的目录路径
        :param prefix: str - 文件名的前缀 (默认: "test4mu_gpu")
        :param file_extension: str - 文件扩展名 (默认: ".json")
        :param target_keys: list or None - 如果指定，提取 JSON 文件中的多个键的值
        :return: list - 包含 (文件名, 键值字典) 的列表
        """
        results = []

        # 检查目录是否存在
        # if not os.path.exists(self.S["directory"]):
        #     raise FileNotFoundError(f"目录 '{self.S["directory"]}' 不存在！")

        # 遍历文件夹中的所有文件
        for filename in os.listdir(self.S["directory"]):
            # 匹配前缀和文件扩展名
            if filename.startswith(self.S["prefix"]) and filename.endswith(self.S["file_extension"]):
                file_path = os.path.join(self.S["directory"], filename)
                try:
                    # 读取 JSON 文件
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)

                    # 如果指定了目标键，提取对应的值
                    extracted_data = {}
                    if self.S["target_keys"]:
                        for key in self.S["target_keys"]:
                            if key in data:
                                extracted_data[key] = data[key]
                            else:
                                print(f"警告: 文件 {filename} 中未找到键 '{key}'")
                    else:
                        extracted_data = data  # 如果未指定键，返回完整数据

                    results.append((filename, extracted_data))

                except json.JSONDecodeError:
                    print(f"错误: 文件 {filename} 不是有效的 JSON 格式")

        return results

class iteration:

    def __init__(self, IN, EV):
        self.S = IN.S
        self.result = self.S | { "time_elapsed": [],
                    "best_ll_nmi": (0,0), "best_ll_acc": (0,0), "best_ll_ari": (0,0), "best_ll_f1": (0,0),
                    "best_ul_nmi": (0,0), "best_ul_acc": (0,0), "best_ul_ari": (0,0), "best_ul_f1": (0,0),
                    "ll_nmi": [], "norm_grad_ll": [], "ll_val": [], "ll_acc": [], "ll_ari":[], "ll_f1": [],
                    "ul_nmi": [], "norm_grad_ul": [], "ul_val": [], "ul_acc": [], "ul_ari": [], "ul_f1": []
                    }

        self.update_learning_rate= True
        self.EV = EV
        self.x = IN.x
        self.y = IN.y
        self.Theta = IN.Theta["LL"]
        self.UL = self.upper_level(self.x, self.y, self.S["lambda_r"])
        self.LL = self.lower_level(self.x, self.y, self.S["lambda_r"], self.Theta)
        self.O = torch.zeros(self.x.shape[0], dtype=torch.float64)
        self.Z = self.O
        self.I = torch.eye(self.x.shape[0], dtype=torch.float64)
        self.grad_x = self.O
        self.grad_y = self.O
        match self.S["clip_method"]:
            case "gaussian":
                self.alpha = lambda x: math.exp(-x**2)
                self.beta = lambda x: math.exp(-x**2)
            case "com":
                self.alpha = lambda x: 1/(x+1)
                self.beta = lambda x: 1/(x+1)
        self.p_u = 0
        self.p_l = 0
        self.ul_val = 0
        self.ll_val = 0

    def syn(self, var:Literal["x","y"]):
        with torch.no_grad():
            match var:
                case "x":
                    self.LL.x.copy_(self.x)
                    self.UL.x.copy_(self.x)
                case "y":
                    self.LL.y.copy_(self.y)
                    self.UL.y.copy_(self.y)

    def get_hessian(self):
        ul_ll_xy = 2 * self.S["lambda_r"] * (self.y @ self.x.T + self.x @ self.y.T)  # 混合二阶偏导
        ul_yy = 2 * self.S["lambda_r"] * self.x @ self.x.T
        ll_yy = 2 * self.Theta + ul_yy

        A = (self.p_u + self.p_l) * ul_ll_xy
        B = self.I + (self.p_u * ul_yy + self.p_l * ll_yy)
        self.Z = B @ self.Z + A

    class lower_level(nn.Module):
        def __init__(self, x, y, lambda_r, Theta):
            super().__init__()
            self.x = nn.Parameter(x)
            self.y = nn.Parameter(y)
            self.lambda_r = lambda_r
            self.Theta = Theta

        def forward(self):
            term1 = torch.trace(self.y.T @ self.Theta @ self.y)
            term2 = self.lambda_r * torch.trace(self.y @ self.y.T @ self.x @ self.x.T)
            # regularization = 1 * torch.norm(self.y, p=2)
            # print((term1+term2)/regularization)
            return (term1 + term2 )# - regularization)

    class upper_level(nn.Module):
        def __init__(self, x, y, lambda_r):
            super().__init__()
            self.x = nn.Parameter(x)
            self.y = nn.Parameter(y)
            self.lambda_r = lambda_r

        def forward(self):
            term1 = self.lambda_r * torch.trace(self.y @ self.y.T @ self.x @ self.x.T)
            # term2 = 0.1 * torch.linalg.norm(self.x)
            return  (term1 )

    def update_value(self, x, grad, method: bool = False):
        x = x + self.S["learning_rate"] * grad
        if method:
            x, _ = torch.linalg.qr(x, mode="reduced")
        return x

    def record_best(self):
        for level in ["ll","ul"]:
            for type in ["acc", "nmi", "ari", "f1"]:
                data = self.result[f"{level}_{type}"]
                max_val = max(data)
                max_index =  data.index(max_val)
                self.result[f"best_{level}_{type}"] = (max_index,max_val)

    @staticmethod
    def Proj(vector, y):
        Proj_LL = torch.eye(y.shape[0]) - y @ y.T
        return Proj_LL @ vector


    def inner_loop(self):
        # 使用向前传播方法计算超梯度
        self.Z = self.O
        for epoch in range(self.S["max_ll_epochs"]):
            self.ll_val = self.LL()
            ll_y = torch.autograd.grad(self.ll_val, self.LL.y, retain_graph=True)[0]

            self.ul_val = self.UL()
            ul_y = torch.autograd.grad(self.ul_val, self.UL.y, retain_graph=True)[0]

            self.p_u = self.S["mu"] * self.alpha(epoch) * self.S["s_u"]
            self.p_l = (1 - self.S["mu"]) * self.beta(epoch) * self.S["s_l"]

            self.grad_y = self.Proj(self.p_u * ul_y + self.p_l * ll_y, self.y)
            self.y = self.update_value(self.y, self.grad_y, self.S["orth_y"])

            self.syn("y")
            self.get_hessian()

        self.ul_val = self.UL()
        ul_x = torch.autograd.grad(self.ul_val, self.UL.x)[0]
        self.ll_val = self.LL()
        ll_x = torch.autograd.grad(self.ll_val, self.LL.x)[0]

        self.grad_x = self.S["lambda_x"] * self.Proj(ul_x + self.Z.T @ ll_x, self.x)
        norm_grad_ll = torch.linalg.norm(self.grad_y, ord=2)
        ll_acc, ll_nmi, ll_ari, ll_f1 = self.EV.assess(self.y.detach().numpy())
        self.EV.record(self.result, "LL", val=self.ll_val.item(), grad=norm_grad_ll.item(),
                       acc=ll_acc, nmi=ll_nmi, ari=ll_ari, f1=ll_f1)


    def outer_loop(self):
        for epoch in range(self.S["max_ul_epochs"]):
            self.x = self.update_value(self.x, self.grad_x, self.S["orth_x"])
            self.ul_val = self.UL()
            self.syn("x")
            self.update_lambda_r()

            ul_acc, ul_nmi, ul_ari, ul_f1 = self.EV.assess(self.x.detach().numpy())
            norm_grad_x = torch.linalg.norm(self.grad_x, ord=2)
            self.EV.record(self.result,"UL", val=self.ul_val.item(), grad=norm_grad_x.item(),
                           acc=ul_acc, nmi=ul_nmi, ari=ul_ari, f1=ul_f1)

    def run(self):
        start_time = time.time()
        for epoch in range(self.S["Epochs"]):
            self.inner_loop()
            self.outer_loop()
            self.result["time_elapsed"].append(time.time() - start_time)

        self.record_best()
        self.EV.use_result(self.result,'dump',self.EV.result_file_name())
        if self.S["result_output"] != "none":
            data = self.EV.use_result({}, "load", self.EV.result_file_name())
            self.EV.plot_result(data, self.S["plot_content"],self.S["result_output"], picname=self.EV.result_fig_name())
        # print(f"ul_nmi:{self.result['best_ul_nmi']}, time: {self.result['time_elapsed'][self.result['best_ul_nmi'][0]]}\n"
        #  f" ul_acc: {self.result['best_ul_acc']}, time: {self.result['time_elapsed'][self.result['best_ul_acc'][0]]}\n"
        #      f"all iteration time: {self.result['time_elapsed'][-1]}")

    def update_lambda_r(self):
        if self.S["update_lambda_r"]:
            val = torch.trace(self.x.T @ (torch.eye(self.x.shape[0]) - self.x @ self.x.T) @ self.x)
            # print(val.item())
            if val <= self.S["epsilon"]:
                self.S["lambda_r"]= self.S["lambda_r"]/2
                return True
            else:
                return False

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

    # 数据集导入以及计算的基本设置
    parser.add_argument('--root_path', type=str, default="/Users/cz/Documents/ML_datasets/3sources",
                        help = "数据集的根目录")
    parser.add_argument('--device_set', type=str2bool, default=True,
                        help = "True 默认使用gpu, cuda或者mps，False 直接使用cpu")
    parser.add_argument('--view_num', type=int, choices=[2,3], default=2,
                        help = "视角个数，数量不同策略不同。")
    parser.add_argument('-v','--view2', type=int, choices=[0,2,4], default=2,
                        help = "双视角时的数据选择，有 0 2 4 共三种")
    parser.add_argument('--seed_num', type=int, default=44,
                        help = "随机种子数，仅数据生成时超图的超边权重处随机。")

    # 迭代次数控制
    parser.add_argument('-L','--max_ll_epochs', type= int, default=10,
                        help = "下层优化函数内部迭代次数")
    parser.add_argument('-U','--max_ul_epochs', type=int, default=1,
                        help = "上层优化函数内部迭代次数")
    parser.add_argument('-E','--Epochs', type=int, default=300,
                        help = "总迭代次数")

    # 聚合时使用的参数
    parser.add_argument('--s_u', type=float, default=0.5,
                        help = "聚合梯度中的上层梯度的系数")
    parser.add_argument('--s_l', type=float, default=0.5, 
                        help = "聚合梯度中的下层梯度的系数")
    parser.add_argument('--mu', type=float, default=0.5, 
                        help = "聚合梯度中的分配系数，范围[0-1]， 取0时为下层梯度，1为上层梯度")
    parser.add_argument('--lambda_x', type=float, default=1, 
                        help = "超梯度的系数")
    parser.add_argument('--update_learning_rate', type=str2bool, default=False,
                        help = "是否更新学习率，默认不更新。")
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help = "更新值的时候使用，在x和y的更新时均使用。")
    parser.add_argument('--orth_y', type=str2bool, default=True,
                        help = "内循环结束是是否正交化y，使用修正的QR分解。")
    parser.add_argument('--orth_x', type=str2bool, default=True,
                        help = "内循环结束是是否正交化x，使用修正的QR分解。")
    parser.add_argument('--clip_method', type=str,choices=["gaussian","com"], default="gaussian",
                        help = "减小{aplha}和{beta}的方法：'gaussian'是高斯函数，'com'是常规反比例函数。")

    # 涉及到聚类部分的参数
    parser.add_argument('--cluster_method', type=str, choices=["spectral", "normal"], default="normal",
                        help = "底层聚类的时候使用谱聚类还是正常聚类，有选择 'spectral'和'normal'")
    parser.add_argument('--update_lambda_r', type=str2bool, default=False,
                        help = "是否更新组合参数")
    parser.add_argument('--lambda_r', type=float, default=1.0, 
                        help = "多视角超图谱聚类的组合参数")
    parser.add_argument('--epsilon', type=float, default=1E-5,
                        help = "多视角超图谱聚类的阈值参数")

    # 结果的处理方式
    parser.add_argument('--result_output', type=str,choices=["show","save","none"], default="none",
                        help = "图片展示的方式，'show' 为输出到窗口，'save'为保存到文件, 'None'为不输出")
    parser.add_argument('--plus_datetime', type=str2bool, default=True,
                        help = "是否在结果文件中添加上时间戳，默认 否")
    parser.add_argument('--file_name', type=str, default="result",
                        help = "输出IT.result中的结果到该文件中，使用json格式，此处不添加扩展名")
    parser.add_argument('--plot_content',type=str, nargs='+', default=["val", "acc", "nmi", "f1"],
                        help="一个列表，包含以下字符串的任意多个, 'grad','val','acc','nmi','ari','f1'")
    parser.add_argument('--figure_name', type=str, default="figure1.png",
                        help = "保存图片为文件的时候，图片的文件名")

    # 结果的批量引用
    parser.add_argument('--directory', type=str, default="logs",
                        help = "结果的默认输出文件夹，无需修改。")
    parser.add_argument('--prefix', type=str, default="test4mu_cpu",
                        help = "选择目标文件夹下特定文件的前缀。")
    parser.add_argument('--file_extension', type=str, default=".json",
                        help = "选择目标文件夹下特征文件的后缀。")
    parser.add_argument('--target_keys', type=str, nargs='+',default=["mu","best_ul_acc"],
                        help = "选择的文件中内容的键值，支持多组。")
    parser.add_argument('--key_x', type=str, default="mu",
                        help = "作图时作为横轴的元素。")
    parser.add_argument('--key_y', type=str, default="best_ul_acc",
                        help = "作图时作为纵轴的元素。")


    S0 = parser.parse_args()
    S = vars(S0)
    return S

def create_instances(S:dict):
    DI = data_importation(S)
    IN = initialization(DI)
    EV = evaluation(DI)

    IT = iteration(IN, EV)
    return IT


