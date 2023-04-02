import os
import sys
from functools import partial
import pandas as pd
import numpy as np
import json
import math
import networkx as nx
import scipy.sparse as sp

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_undirected, coalesce, from_networkx


class MUSAE(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'


class Twitch(InMemoryDataset):
    url = 'https://graphmining.ai/datasets/twitch'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        assert self.name in ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        edges = 'musae_{}_edges.csv'.format(self.name)
        if self.name == 'DE':
            features = 'musae_{}.json'.format(self.name)
        else:
            features = 'musae_{}_features.json'.format(self.name)
        target = 'musae_{}_target.csv'.format(self.name)
        return [edges, features, target]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = download_url(f'{self.url}/{self.name}.zip', self.raw_dir)
        extract_zip(path, self.raw_dir)
        # The zip file is removed
        os.unlink(path)

    def process(self):
        feature_file = os.path.join(self.raw_dir, self.name, self.raw_file_names[1])
        with open(feature_file) as f:
            data = json.load(f)
            num_nodes = len(data)
            num_features = 3170

            x = np.zeros((num_nodes, num_features))
            for idx, elem in data.items():
                x[int(idx), elem] = 1
            x = torch.from_numpy(np.array(x)).float()

        target_file = os.path.join(self.raw_dir, self.name, self.raw_file_names[2])
        data = pd.read_csv(target_file)
        mature = list(map(int, data['mature'].values))
        new_id = list(map(int, data['new_id'].values))
        idx_map = {elem: i for i, elem in enumerate(new_id)}
        y = [mature[idx_map[idx]] for idx in range(num_nodes)]
        y = torch.from_numpy(np.array(y))

        edge_file = os.path.join(self.raw_dir, self.name, self.raw_file_names[0])
        edges = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edges.to_numpy()).to(torch.long)
        edge_index = edge_index.t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Twitch-{self.name}()'


class WebKB(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root, transform=None, pre_transform=None):
        self.univ_list = ['cornell', 'texas', 'wisconsin']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def raw_file_names(self):
        out = ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
        return out

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.univ_list:
            for f in self.raw_file_names:
                download_url(f'{self.url}/new_data/{name}/{f}', os.path.join(self.raw_dir, name))

    def read_web_kb(self, name):
        out = ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

        raw_paths = [os.path.join(self.raw_dir, name, filename) for filename in out]

        with open(raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.name = name

        return data

    def process(self):
        data_list = []

        for name in self.univ_list:
            data = self.read_web_kb(name)
            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate([data_list]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Student(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        pkl_file = '{}_synthetic_graph.pkl'.format(self.name)

        return pkl_file

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_file = os.path.join(self.raw_dir, self.raw_file_names)
        G = nx.read_gpickle(data_file)

        attrs = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                 'health', 'absences', 'G1', 'G2', 'class']

        g = G.to_networkx()
        data = from_networkx(g, group_node_attrs=attrs)
        y = data.x[:, -1:].squeeze().to(torch.long)
        x = data.x[:, :-1].to(torch.float)
        edge_index = data.edge_index
        attrs = attrs[:-1]

        df = pd.DataFrame(x, columns=attrs)
        selected_attrs = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
                          'reason', 'guardian', 'traveltime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                          'nursery', 'higher', 'internet', 'romantic', 'goout', 'G1', 'G2']
        df = df[selected_attrs]

        g1 = df['G1'] >= 10
        df['G1'] = list(map(int, g1))

        g2 = df['G2'] >= 10
        df['G2'] = list(map(int, g2))

        # absences = df['absences'] >= 5
        # df['absences'] = list(map(int, absences))

        df['age'].replace(20, 19, inplace=True)

        feature_to_encode = ['age', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'failures',
                             'goout']
        dummies = pd.get_dummies(df[feature_to_encode], columns=feature_to_encode)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(feature_to_encode, axis=1)

        x = torch.from_numpy(df.to_numpy()).to(torch.float)
        num_nodes = len(y)
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Synthetic-{self.name}()'


class German(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        pkl_file = '{}_synthetic_graph.pkl'.format(self.name)

        return pkl_file

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_file = os.path.join(self.raw_dir, self.raw_file_names)
        G = nx.read_gpickle(data_file)

        attrs = ['Gender', 'ForeignWorker', 'Single', 'Age', 'LoanDuration', 'LoanAmount', 'LoanRateAsPercentOfIncome',
                 'YearsAtCurrentHome', 'NumberOfOtherLoansAtBank', 'NumberOfLiableIndividuals', 'HasTelephone',
                 'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100',
                 'SavingsAccountBalance_geq_500', 'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere',
                 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse',
                 'Unemployed', 'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled',
                 'PurposeOfLoan_Business', 'PurposeOfLoan_Education', 'PurposeOfLoan_Electronics',
                 'PurposeOfLoan_Furniture', 'PurposeOfLoan_HomeAppliances', 'PurposeOfLoan_NewCar',
                 'PurposeOfLoan_Other',
                 'PurposeOfLoan_Repairs', 'PurposeOfLoan_Retraining', 'PurposeOfLoan_UsedCar', 'class']

        g = G.to_networkx()
        data = from_networkx(g, group_node_attrs=attrs)
        y = data.x[:, -1:].squeeze().to(torch.long)
        x = data.x[:, :-1].to(torch.float)
        edge_index = data.edge_index
        attrs = attrs[:-1]

        df = pd.DataFrame(x, columns=attrs)
        selected_attrs = ['Gender', 'ForeignWorker', 'Single', 'Age', 'LoanDuration', 'LoanAmount',
                          'LoanRateAsPercentOfIncome',
                          'YearsAtCurrentHome', 'NumberOfOtherLoansAtBank', 'NumberOfLiableIndividuals', 'HasTelephone',
                          'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200',
                          'SavingsAccountBalance_geq_100',
                          'SavingsAccountBalance_geq_500', 'MissedPayments', 'NoCurrentLoan',
                          'CriticalAccountOrLoansElsewhere',
                          'OtherLoansAtBank', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse',
                          'Unemployed', 'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled',
                          'PurposeOfLoan_Business', 'PurposeOfLoan_Education', 'PurposeOfLoan_Electronics',
                          'PurposeOfLoan_Furniture', 'PurposeOfLoan_HomeAppliances', 'PurposeOfLoan_NewCar',
                          'PurposeOfLoan_Other',
                          'PurposeOfLoan_Repairs', 'PurposeOfLoan_Retraining', 'PurposeOfLoan_UsedCar']
        df = df[selected_attrs]

        df['NumberOfLiableIndividuals'] = df['NumberOfLiableIndividuals'] - 1

        df.loc[df['Age'] <= 30., 'Age'] = 0
        df.loc[((df['Age'] > 30.) & (df['Age'] <= 50.)), 'Age'] = 30
        df.loc[df['Age'] > 50., 'Age'] = 50

        df.loc[df['LoanDuration'] <= 10., 'LoanDuration'] = 0
        df.loc[((df['LoanDuration'] > 10.) & (df['LoanDuration'] <= 20.)), 'LoanDuration'] = 10
        df.loc[((df['LoanDuration'] > 20.) & (df['LoanDuration'] <= 30.)), 'LoanDuration'] = 20
        df.loc[df['LoanDuration'] > 30., 'LoanDuration'] = 30

        df.loc[df['LoanAmount'] <= 1000., 'LoanAmount'] = 0
        df.loc[((df['LoanAmount'] > 1000.) & (df['LoanAmount'] <= 2000.)), 'LoanAmount'] = 1000
        df.loc[((df['LoanAmount'] > 2000.) & (df['LoanAmount'] <= 3000.)), 'LoanAmount'] = 2000
        df.loc[((df['LoanAmount'] > 3000.) & (df['LoanAmount'] <= 4000.)), 'LoanAmount'] = 3000
        df.loc[df['LoanAmount'] > 4000., 'LoanAmount'] = 4000

        g1 = df['LoanRateAsPercentOfIncome'] >= 3
        df['LoanRateAsPercentOfIncome'] = list(map(int, g1))

        g1 = df['YearsAtCurrentHome'] >= 3
        df['YearsAtCurrentHome'] = list(map(int, g1))

        g1 = df['NumberOfOtherLoansAtBank'] > 1
        df['NumberOfOtherLoansAtBank'] = list(map(int, g1))

        feature_to_encode = ['Age', 'LoanDuration', 'LoanAmount']
        dummies = pd.get_dummies(df[feature_to_encode], columns=feature_to_encode)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(feature_to_encode, axis=1)

        x = torch.from_numpy(df.to_numpy()).to(torch.float)
        num_nodes = len(y)
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Synthetic-{self.name}()'


def load_dataset(dataset, val_ratio=.25, test_ratio=.25, cols_to_group=0):
    path = os.path.join('./datasets', dataset)
    if dataset in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(path, dataset)
    elif dataset in ['dblp']:
        data = CitationFull(path, dataset)
    elif dataset in ['facebook', 'deezer', 'github', 'wikipedia']:
        data = MUSAE(path, dataset)
    elif dataset.startswith('twitch'):
        identifier = dataset[dataset.find('/') + 1:]
        data = Twitch(path, identifier)
    elif dataset in ['student']:
        data = Student(path, dataset)
    elif dataset in ['german']:
        data = German(path, dataset)
    else:
        print("dataset not supported")
        sys.exit(1)

    data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
    if dataset in ['pubmed']:
        data.x[data.x > 0] = 1
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1
    data.y = torch.nn.functional.one_hot(data.y, num_classes=data.num_classes).float()
    data = preprocess_features(data, cols_to_group)

    return data


def preprocess_features(data, cols_to_group=1):
    if cols_to_group is not None and cols_to_group > 1:
        d = data.x.shape[1]
        d_prime = math.ceil(d / cols_to_group)
        index = 0

        x_new = torch.zeros((data.x.shape[0], d_prime)).to(data.x.device)
        i = 0

        while index < data.x.shape[1]:
            if index + cols_to_group < data.x.shape[1]:
                x_new[:, i] = data.x[:, index:index + cols_to_group].sum(dim=1)
            else:
                x_new[:, i] = data.x[:, index:].sum(dim=1)
            i = i + 1
            index = index + cols_to_group

        data.x = x_new
        data.x[data.x > 0] = 1

        data.num_features = data.x.shape[1]

    return data


