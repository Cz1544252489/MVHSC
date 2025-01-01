# Data importation
import os
import zipfile
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
    def __init__(self):
        self.device = torch.device("cpu")
        self.dataset_download_or_not = False
        self.dataset_name = "3sources"
        self.root_path = "./"
        self.dataset_link = "http://mlg.ucd.ie/files/datasets/3sources.zip"
        self.view_num = None
        self.cluster_num = None
        self.sources = None
        self.file_types = None
        self.label_mapping = None
        self.loop_mapping = None
        self.view = None
        self.pre_data_definition()
        np.random.seed()
        if self.dataset_download_or_not:
            self.download_dataset()
        self.splited_data = self.split_data()
        self.Theta, self.x, self.y = self.inital()

    def pre_data_definition(self):
        match self.dataset_name:
            case "3sources":
                self.sources = ['bbc', 'guardian', 'reuters']
                self.view_num = 2
                self.view = 2
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

        # 创建DataFrame
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
                    l += 1
        return Theta, F["UL"], F["LL"]



