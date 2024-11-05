# This file aims to collect the auxiliary function of MVHSC.

import os


os.environ["OMP_NUM_THREADS"] = "1"

from typing import Literal
import pandas as pd
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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

np.random.seed(42)
# function part

# data importation

class data_importation:
    def __init__(self, view_num:Literal[1,2,3]=2, view2:Literal[0,2,4] = 0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.view_num = view_num
        self.view2 = view2
        self.device = device
        self.cluster_num = 6
        self.root_path = "E:\\DL_datasets\\3sources\\"
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



class initialization():

    def __init__(self, DI):
        self.mapping2 = ["UL","LL"]
        self.mapping1 = DI.mapping1
        self.view2 = DI.view2
        self.data = DI.data
        self.device = DI.device
        self.Theta, self.F = self.initial(self.data["view_num"])

    @staticmethod
    def row_norm(X):
        row_norms = np.linalg.norm(X, axis = 1)
        X_normalized = X / row_norms
        return X_normalized

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

        return Theta, F

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


class clustering():

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

    def cluster(self, F, num_type, method:Literal["normal","spectral"] ="normal", show=False):
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

class iteration():

    def __init__(self, EV, IN, settings):
        self.grad_method = "man"
        self.learning_rate = settings["learning_rate"]
        self.lambda_r = settings["lambda_r"]
        self.result = {"ll_nmi": [], "norm_grad_ll": [], "ll_val": [],
                  "ul_nmi": [], "norm_grad_ul": [], "ul_val": [],
                  "best_ll_nmi": 0, "best_ul_nmi": 0}
        self.orth1 = settings["orth1"]
        self.orth2 = settings["orth2"]
        self.update_lambda_or_not = settings["update_lambda"]
        self.epsilon = settings["epsilon"]
        self.update_learning_rate= True
        self.EV = EV
        self.F = IN.F
        self.Theta = IN.Theta
        self.max_ll_epochs = settings["max_ll_epochs"]
        self.max_ul_epochs = settings["max_ul_epochs"]
        self.UL = self.upper_level(self.F["UL"])
        self.LL = self.lower_level(self.F["LL"])

    class lower_level(nn.Module):
        def __init__(self, F_ll):
            # 变量是F_LL也是原来的F
            super().__init__()

            self.F_ll = nn.Parameter(F_ll)

        def forward(self, F_ul, Theta, lambda_r):
            term1 = torch.trace(self.F_ll.T @ Theta @ self.F_ll)
            term2 = lambda_r * torch.trace(self.F_ll @ self.F_ll.T @ F_ul @ F_ul.T)
            return -(term1 + term2)

    class upper_level(nn.Module):
        def __init__(self, F_ul):
            super().__init__()
            self.F_ul = nn.Parameter(F_ul)

        def forward(self, F_ll, lambda_r):
            term = lambda_r * torch.trace(F_ll @ F_ll.T @ self.F_ul @ self.F_ul.T)
            return -term

    @staticmethod
    def update_value(F, grad_F, learning_rate, method: bool = True):
        F += learning_rate * grad_F
        if method:
            F, _ = torch.linalg.qr(F, mode="reduced")
        return F

    def inner_loop(self):
        for epoch in range(self.max_ll_epochs):
            LL_val = self.LL(self.F["UL"], self.Theta["LL"], self.lambda_r)
            match self.grad_method:
                case "man":
                    Theta_ = self.Theta["LL"] + self.lambda_r * self.F["UL"] @ self.F["UL"].T
                    # Proj_ = torch.eye(F["LL"].shape[0]) - F["LL"] @ F["LL"].T
                    grad_ll = 2 * Theta_ @ self.F["LL"]

            try:
                grad_ll
            except NameError:
                print("grad_ll未定义")

            self.F["LL"] = self.update_value(self.F["LL"], grad_ll, self.learning_rate, self.orth1)
            ll_nmi, _ = self.EV.assess(self.F["LL"])
            if ll_nmi > self.result["best_ll_nmi"]:
                self.result["best_ll_nmi"] = ll_nmi
                self.result["best_F_ll"] = self.F["LL"].tolist()
            norm_grad_ll = torch.linalg.norm(grad_ll, ord =2).item()

        self.F["LL"] = self.update_value(self.F["LL"], grad_ll,0, self.orth2)
        self.EV.record(self.result, LL_val.item(), ll_nmi, norm_grad_ll,"LL")

    def outer_loop(self):
        for epoch in range(self.max_ul_epochs):
            UL_val = self.UL(self.F["LL"], self.lambda_r)
            match self.grad_method:
                case "man":
                    Theta_ = self.lambda_r * self.F["LL"] @ self.F["LL"].T
                    # Proj_ = torch.eye(F["LL"].shape[0]) #  - F["UL"] @ F["UL"].T
                    grad_ul = 2 * Theta_ @ self.F["UL"]

            try:
                grad_ul
            except NameError:
                print("grad_ul未定义")

            self.F["UL"] = self.update_value(self.F["UL"], grad_ul, self.learning_rate, self.orth1)
            ul_nmi, _ = self.EV.assess(self.F["UL"])
            if ul_nmi > self.result["best_ul_nmi"]:
                self.result["best_ul_nmi"] = ul_nmi
                self.result["best_F_ul"] = self.F["UL"].tolist()
            norm_grad_ul = torch.linalg.norm(grad_ul, ord =2).item()

        self.F["UL"] = self.update_value(self.F["UL"], grad_ul, 0, self.orth2)
        self.EV.record(self.result, UL_val.item(), ul_nmi, norm_grad_ul, "UL")

    def update_lambda(self):
        if self.update_lambda_or_not:
            val = torch.trace(self.F["UL"].T @ (torch.eye(self.F["UL"].shape[0]) - self.F["LL"] @ self.F["LL"].T) @ self.F["UL"])
            if val <= self.epsilon:
                self.lambda_r = self.lambda_r /2
            return val

class evaluation():

    def __init__(self, DI, CL):
        self.cluster = CL.cluster
        self.data = DI.data
        self.mapping1 = DI.mapping1
        self.view2 = DI.view2
        mapping = {
            "business": 1,
            "entertainment": 2,
            "health": 3,
            "politics": 4,
            "sport": 5,
            "tech": 0
        }
        self.mapping = mapping

    def assess(self, data1):
        i,j = self.mapping1[self.view2][0:2]
        sources = self.data["sources"]
        labels_pred = self.cluster(data1, self.data["cluster_num"])
        labels_true = self.data[f"labels_true_{sources[i]}_{sources[j]}"]
        nmi = self.calculate_nmi(labels_true, labels_pred)
        ari = self.calculate_ari(labels_true, labels_pred)
        return nmi, ari

    def replace(self, labels):
        replaced_list = [self.mapping[item] for item in labels]
        return replaced_list

    def calculate_nmi(self, labels_true, labels_pred, method:Literal["min","geometric","arithmetic","max"]="min"):
        # 初步看，使用"min" 参数有好的结果
        return normalized_mutual_info_score(labels_true, labels_pred,average_method=method)

    def calculate_ari(self, labels_true, labels_pred):
        return adjusted_rand_score(labels_true, labels_pred)

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
    def record(result, val, nmi, norm_grad, type:Literal["UL","LL"]):
        match type:
            case "UL":
                result["ul_val"].append(val)
                result["ul_nmi"].append(nmi)
                result["norm_grad_ul"].append(norm_grad)
            case "LL":
                result["ll_val"].append(val)
                result["ll_nmi"].append(nmi)
                result["norm_grad_ll"].append(norm_grad)
        return result

    def plot_result(self,data, flag):
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
        plt.show()

    @staticmethod
    def use_result(data, method: Literal["dump","load"]):
        file_name = "result.json"
        match method:
            case "dump":
                with open(file_name, "w") as file:
                    json.dump(data, file, indent=4)
            case "load":
                with open(file_name, "r") as file:
                    data = json.load(file)
                return data


    @staticmethod
    def output_type(result, flag):
        output = {}
        if "nmi" in flag :
            output["ll_nmi"] = result["ll_nmi"]
            output["ul_nmi"] = result["ul_nmi"]
        if "val" in flag:
            output["ll_val"] =result["ll_val"]
            output["ul_val"] =result["ul_val"]
        if "grad" in flag:
            output["norm_grad_ll"] =  result["norm_grad_ll"]
            output["norm_grad_ul"] = result["norm_grad_ul"]

        return output

def create_instances(settings, view2=0):
    DI = data_importation(view2 = view2)
    IN = initialization(DI)
    CL = clustering()
    EV = evaluation(DI, CL)
    IT = iteration(EV, IN, settings)
    return DI, IN, CL, EV, IT

class test_part():

    def __init__(self, DI, CL, EV, IN):
        self.file_name = "output.txt"
        self.sources = DI.sources
        self.view_num = DI.view_num
        self.data = DI.data
        self.CL = CL
        self.EV = EV
        self.IN = IN

    def output_result(self, input, output_method:Literal["file","console"]="console"):
        match output_method:
            case "file":
                with open(self.file_name, 'a') as file:
                    print(input, file=file)
            case "console":
                print(input)


    def cluster_and_evaluation(self ,method:Literal["file","console"]="console"):
        data = self.data
        sources = self.sources
        p = data["cluster_num"]
        match self.view_num:
            case 1:
                for view in sources:
                    labels_pred = self.CL.cluster(data[f'{view}_mtx'], p, method="spectral")
                    nmi = self.EV.calculate_nmi(data[f"{view}_labels_true"], labels_pred)
                    self.output_result(f"SC_{view}:{nmi}", method)

                for view in sources:
                    _, F = self.IN.get_Theta_and_F(data[f'{view}_mtx'], p)
                    labels_pred = self.CL.cluster(F, p, method="spectral")
                    nmi = self.EV.calculate_nmi(data[f"{view}_labels_true"], labels_pred)
                    self.output_result(f"HSC_{view}:{nmi}", method)

                self.output_result("-"*30, method)

            case 2:

                for i, j, k in [[0,1,0],[0,1,1],[0,2,0],[0,2,2],[1,2,1],[1,2,2]]:
                # for i,j in [[0,1],[0,2],[1,2]]:
                    labels_pred = self.CL.cluster(data[f"{sources[k]}_mtx_{sources[i]}_{sources[j]}"], p, method="spectral")
                    nmi = self.EV.calculate_nmi(data[f"labels_true_{sources[i]}_{sources[j]}"], labels_pred)
                    self.output_result(f"{sources[k]}_SC_{sources[i]}_{sources[j]}:{nmi}", method)
                    self.output_result("-"*30, method)

                for i, j, k in [[0,1,0],[0,1,1],[0,2,0],[0,2,2],[1,2,1],[1,2,2]]:
                # for i,j in [[0,1],[0,2],[1,2]]:
                     _, F = self.IN.get_Theta_and_F(data[f'{sources[k]}_mtx_{sources[i]}_{sources[j]}'], p)
                     labels_pred = self.CL.cluster(F, p, method="spectral")
                     nmi = self.EV.calculate_nmi(data[f"labels_true_{sources[i]}_{sources[j]}"], labels_pred)
                     self.output_result(f"{sources[k]}_HSC_{sources[i]}_{sources[j]}:{nmi}", method)
                     self.output_result("-"*30, method)

            case 3:
                for i in range(3):
                    labels_pred = self.CL.cluster(data[f"{sources[i]}_mtx_3sources"],p)
                    nmi = self.EV.calculate_nmi(data[f"labels_true_3sources"], labels_pred)
                    self.output_result(f"{sources[i]}_SC_3sources:{nmi}", method)
                    self.output_result("-"*30, method)

                for i in range(3):
                    _, F = self.IN.get_Theta_and_F(data[f'{sources[i]}_mtx_3sources'], p)
                    labels_pred = self.CL.cluster(F, p, method="spectral")
                    nmi = self.EV.calculate_nmi(data[f"labels_true_3sources"], labels_pred)
                    self.output_result(f"{sources[i]}_HSC_3sources:{nmi}", method)
                    self.output_result("-"*30, method)



