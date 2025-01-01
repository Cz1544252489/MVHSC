# Data importation
import os
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix


class data_importation:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dataset_download_or_not = False
        self.dataset_name = "3sources"
        self.root_path = "./"
        self.dataset_link = "http://mlg.ucd.ie/files/datasets/3sources.zip"
        self.sources = None
        self.file_types = None
        self.label_mapping = None
        self.loop_mapping = None
        self.pre_data_definition()
        np.random.seed()
        if self.dataset_download_or_not:
            self.download_dataset()

    def pre_data_definition(self):
        match self.dataset_name:
            case "3sources":
                self.sources = ['bbc', 'guardian', 'reuters']
                self.file_types = ['mtx', 'terms', 'docs']
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
