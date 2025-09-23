import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

class IBM_Dataset:
    def __init__(self):
        self.transactions_df = None
        self.users_df = None
        self.cards_df = None

        self.edges = None

        self.node_user_cards = None
        self.node_merchants = None

        self.node_to_idx = dict()
        self.idx_to_node = dict()
        self.mcc_to_idx = dict()
        self.idx_to_mcc = dict()
        self.zip_to_idx = dict()
        self.idx_to_zip = dict()

        self.node_attr = None
        self.x = None
        self.node_zip_indices = None

    def read_transactions_csv(self, file_path):
        print("Loading transactions CSV...")
        self.transactions_df = pd.read_csv(file_path)
        print("Transactions CSV loaded successfully.")
        return self

    def read_users_csv(self, file_path):
        self.users_df = pd.read_csv(file_path)
        print("Users CSV loaded successfully.")
        return self

    def read_cards_csv(self, file_path):
        self.cards_df = pd.read_csv(file_path)
        print("Cards CSV loaded successfully.")
        return self

    def preprocess_transactions(self):
        if self.transactions_df is None:
            raise ValueError("Transactions dataframe is not loaded. Please call read_transactions_csv() first.")

        print("Preprocessing transactions data...")

        self.edges = self.transactions_df.copy()

        # Create a 'DateTime' column
        self.edges['DateTime'] = pd.to_datetime(
            self.edges['Year'].astype(str) + '-' +
            self.edges['Month'].astype(str).str.zfill(2) + '-' +
            self.edges['Day'].astype(str).str.zfill(2) + ' ' +
            self.edges['Time']
        )

        # Create a separate 'Date' column
        self.edges['Date'] = pd.to_datetime(
            self.edges['Year'].astype(str) + '-' +
            self.edges['Month'].astype(str).str.zfill(2) + '-' +
            self.edges['Day'].astype(str).str.zfill(2)
        )

        self.edges = self.edges.sort_values(by='DateTime', ascending=True)
        # self.edges = self.edges[self.edges['DateTime'] < '2020-01-01']
        self.edges = self.edges.reset_index(drop=True)

        self.edges['isFraud'] = self.edges['Is Fraud?'].map({'No': 0, 'Yes': 1})

        # Select relevant columns
        self.edges = self.edges[['DateTime', 'Date', 'User', 'Card', 'Merchant Name', 'Amount', 'Use Chip', 'Zip', 'MCC', 'Errors?', 'isFraud']]

        self.edges = self.edges.rename(columns={'Errors?': 'Error'})
        self.edges['User_Card'] = self.edges['User'].astype(str) + '_' + self.edges['Card'].astype(str)

        self.edges.loc[:, 'Amount'] = self.edges['Amount'].replace({"\$": "", ",": ""}, regex=True).astype(float)
        self.edges['Src'] = np.where(self.edges['Amount'] >= 0, self.edges['User_Card'], self.edges['Merchant Name']).astype(str)
        self.edges['Dest'] = np.where(self.edges['Amount'] >= 0, self.edges['Merchant Name'], self.edges['User_Card']).astype(str)
        
        # Amount scaling
        scaler = MinMaxScaler()
        abs_amounts = self.edges[['Amount']].abs().astype(float)
        log_scaled_amounts = np.log1p(abs_amounts)
        self.edges['Scaled_Amount'] = scaler.fit_transform(log_scaled_amounts)

        # MCC indexing
        all_mcc = self.edges['MCC'].unique()
        self.mcc_to_idx = {mcc: idx for idx, mcc in enumerate(all_mcc)}
        self.idx_to_mcc = {idx: mcc for mcc, idx in self.mcc_to_idx.items()}
        self.edges['MCC_idx'] = self.edges['MCC'].map(self.mcc_to_idx)

        # Zip code indexing
        self.edges['Zip'] = self.edges['Zip'].fillna(0).astype(float)
        all_zips = self.edges['Zip'].unique()
        self._set_zip_mappings(all_zips)
        self.edges['Zip_idx'] = self.edges['Zip'].map(self.zip_to_idx)

        # Use Chip, Error to one-hot encoding
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        use_chip_onehot = onehot_encoder.fit_transform(self.edges[['Use Chip']])
        use_chip_types = onehot_encoder.get_feature_names_out(['Use Chip'])
        use_chip_df = pd.DataFrame(use_chip_onehot, columns=use_chip_types, index=self.edges.index)

        self.edges['Error'] = self.edges['Error'].fillna('NaN')
        errors = self.edges['Error'].unique()
        error_types = set()
        for errs in errors:
            if errs == 'NaN':
                continue
            for err in errs.split(','):
                error_types.add("Error_" + err.strip())
        error_df = pd.DataFrame(0, columns=list(error_types), index=self.edges.index, dtype=float)
        for i, errs in enumerate(self.edges['Error']):
            if errs == 'NaN':
                continue
            for err in errs.split(','):
                error_df.at[i, "Error_" + err.strip()] = 1

        self.edges = pd.concat([self.edges, use_chip_df, error_df], axis=1)

        # Node Merchants
        merchant_names = self.edges['Merchant Name'].unique()
        merchant_zips = self.edges.groupby('Merchant Name')['Zip_idx'].first().to_dict()
        self.node_merchants = pd.DataFrame({
            'Merchant Name': merchant_names,
            'Zip_idx': [merchant_zips[name] for name in merchant_names]
        })
        self.node_merchants['Merchant Name'] = self.node_merchants['Merchant Name'].astype(str)

        # Drop intermediate columns
        self.edges = self.edges.drop(columns=['DateTime', 'User', 'Card', 'Merchant Name', 'Amount', 'Use Chip', 'Zip', 'MCC', 'Error', 'User_Card'])

        print("Preprocessing transactions completed.")

        return self
    
    def preprocess_user_cards(self):
        if self.users_df is None:
            raise ValueError("Users dataframe is not loaded. Please call read_users_csv() first.")
        if self.cards_df is None:
            raise ValueError("Cards dataframe is not loaded. Please call read_cards_csv() first.")
        
        # User ID indexing
        self.users_df['User'] = self.users_df.index

        # Zip code indexing
        all_zips = self.users_df['Zipcode'].unique()
        self._set_zip_mappings(all_zips)
        self.users_df['Zip_idx'] = self.users_df['Zipcode'].map(self.zip_to_idx)

        # FICP Score categorization
        def categorize_fico(score):
            if score >= 800:
                return 'Excellent'
            elif score >= 740:
                return 'Very Good'
            elif score >= 670:
                return 'Good'
            elif score >= 580:
                return 'Fair'
            else:
                return 'Poor'
        self.users_df['FICO_Category'] = self.users_df['FICO Score'].apply(categorize_fico)

        # select relevant columns
        self.users_df = self.users_df[['User', 'Zip_idx', 'FICO_Category']]

        # Create a 'User_Card' column
        self.cards_df['User_Card'] = self.cards_df['User'].astype(str) + "_" + self.cards_df['CARD INDEX'].astype(str)
        # Select relevant columns
        self.cards_df = self.cards_df[['User', 'User_Card', 'Card Brand', 'Card Type']]

        # Merge users and cards dataframes
        # Node Cards
        self.node_cards = self.users_df.merge(self.cards_df, left_on='User', right_on='User', how='left')

        # One-hot encoding for categorical features
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        card_brand_onehot = onehot_encoder.fit_transform(self.node_cards[['Card Brand']])
        card_brand_types = onehot_encoder.get_feature_names_out(['Card Brand'])
        card_brand_df = pd.DataFrame(card_brand_onehot, columns=card_brand_types, index=self.node_cards.index)
        card_type_onehot = onehot_encoder.fit_transform(self.node_cards[['Card Type']])
        card_type_types = onehot_encoder.get_feature_names_out(['Card Type'])
        card_type_df = pd.DataFrame(card_type_onehot, columns=card_type_types, index=self.node_cards.index)
        fico_onehot = onehot_encoder.fit_transform(self.node_cards[['FICO_Category']])
        fico_types = onehot_encoder.get_feature_names_out(['FICO_Category'])
        fico_df = pd.DataFrame(fico_onehot, columns=fico_types, index=self.node_cards.index)
        self.node_cards = pd.concat([self.node_cards, card_brand_df, card_type_df, fico_df], axis=1)
        self.node_cards = self.node_cards.drop(columns=['Card Brand', 'Card Type', 'FICO_Category'])

        return self
        
    def _set_zip_mappings(self, all_zips):
        for zip in all_zips:
            if zip not in self.zip_to_idx:
                idx = len(self.zip_to_idx)
                self.zip_to_idx[zip] = idx
                self.idx_to_zip[idx] = zip

    def create_node_mappings(self):
        if self.edges is None:
            raise ValueError("Transactions dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")

        src_nodes = self.edges['Src'].unique()
        dest_nodes = self.edges['Dest'].unique()
        all_nodes = np.unique(np.concatenate((src_nodes, dest_nodes)))
        print(f"Total unique nodes: {len(all_nodes)}")
        
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        return self

    def create_graph_x(self):
        if self.node_cards is None:
            raise ValueError("Node cards dataframe is not created. Please call preprocess_user_cards() first.")
        if self.node_merchants is None:
            raise ValueError("Node merchants dataframe is not created. Please call preprocess_transactions() first.")
        if not self.node_to_idx:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")

        all_nodes_df = pd.DataFrame(list(self.node_to_idx.keys()), columns=['Node'])
        user_cards_for_merge = self.node_cards.rename(columns={'User_Card': 'Node'}).copy()
        merchants_for_merge = self.node_merchants.rename(columns={'Merchant Name': 'Node'}).copy()

        self.node_attr = pd.merge(all_nodes_df, user_cards_for_merge, on='Node', how='left')
        self.node_attr = pd.merge(self.node_attr, merchants_for_merge, on='Node', how='left', suffixes=('_user', '_merchant'))

        self.node_attr['Is_User'] = self.node_attr['Node'].isin(self.node_cards['User_Card']).astype(int)
        self.node_attr['Is_Merchant'] = self.node_attr['Node'].isin(self.node_merchants['Merchant Name']).astype(int)
        self.node_attr['Zip_idx'] = np.where(self.node_attr['Is_User'] == 1,
                                                 self.node_attr['Zip_idx_user'],
                                                 self.node_attr['Zip_idx_merchant'])

        attr_cols = ['Is_User', 'Is_Merchant', 'Zip_idx']
        attr_cols += [col for col in self.node_attr.columns if col.startswith('Card Brand_')]
        attr_cols += [col for col in self.node_attr.columns if col.startswith('Card Type_')]
        attr_cols += [col for col in self.node_attr.columns if col.startswith('FICO_Category_')]
        self.node_attr = self.node_attr[attr_cols].fillna(0)

        self.x = torch.tensor(self.node_attr.values, dtype=torch.float)
        self.node_zip_indices = torch.tensor(self.node_attr['Zip_idx'].values, dtype=torch.long)

        return self

    def build_pyg_graph(self, start_date=None, end_date=None):
        if self.edges is None:
            raise ValueError("Edges dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")
        if not self.node_to_idx:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        if self.x is None:
            raise ValueError("Node features are not created. Please call create_graph_x() first.")
        
        filtered_edges = self.get_edges(start_date, end_date)
        src_idx = filtered_edges['Src'].map(self.node_to_idx).to_numpy()
        dest_idx = filtered_edges['Dest'].map(self.node_to_idx).to_numpy()
        edge_index = torch.tensor(np.vstack((src_idx, dest_idx)), dtype=torch.long)

        drop_cols = {'Date', 'Src', 'Dest', 'isFraud', 'MCC_idx', 'Zip_idx'}
        edge_feature_cols = [col for col in filtered_edges if col not in drop_cols]
        edge_attr = torch.tensor(filtered_edges[edge_feature_cols].values, dtype=torch.float)

        edge_mcc_indices = torch.tensor(filtered_edges['MCC_idx'].values, dtype=torch.long)
        edge_zip_indices = torch.tensor(filtered_edges['Zip_idx'].values, dtype=torch.long)

        data = Data(x=self.x, edge_index=edge_index, edge_attr=edge_attr)

        data.node_zip_indices = self.node_zip_indices
        data.edge_mcc_indices = edge_mcc_indices
        data.edge_zip_indices = edge_zip_indices

        data.edge_labels = torch.tensor(filtered_edges['isFraud'].values, dtype=torch.long)

        return data

    def show_graph(self, data):
        if self.node_attr is None:
            raise ValueError("Node attributes are not created. Please call create_graph_x() first.")

        G_vis = to_networkx(data, to_undirected=False, remove_self_loops=False, to_multi=True)
        G_vis = nx.MultiDiGraph(G_vis)  

        # 연결된 노드만 추출
        connected_nodes = set()
        for u, v in G_vis.edges():
            connected_nodes.add(u)
            connected_nodes.add(v)
        G_vis = G_vis.subgraph(connected_nodes).copy()

        # 노드 타입에 따른 색상 지정
        is_user = self.node_attr['Is_User'].values.astype(bool)
        is_merchant = self.node_attr['Is_Merchant'].values.astype(bool)

        node_colors = []
        for node in G_vis.nodes():
            idx = node
            if idx < len(is_user) and is_user[idx]:
                node_colors.append('dodgerblue')  # User_Card 노드 색상
            elif idx < len(is_merchant) and is_merchant[idx]:
                node_colors.append('orange')      # Merchant 노드 색상
            else:
                node_colors.append('gray')        # 기타 노드 색상
        
        # edge 속성에 따른 색상 지정
        edge_labels = data.edge_labels.numpy()
        edge_colors = []
        for i, (u, v, k) in enumerate(G_vis.edges(keys=True)):
            if i < len(edge_labels) and edge_labels[i] == 1:
                edge_colors.append('red')  # 사기 거래 색상
            else:
                edge_colors.append('lightgray')  # 정상 거래 색상
        
        # 그래프 그리기
        plt.figure(figsize=(12, 8))

        # 중복 엣지 시 곡선 처리
        pos = nx.spring_layout(G_vis, seed=42)
        rad_map = {}
        for i, (u, v, k) in enumerate(G_vis.edges(keys=True)):
            rad = rad_map.get((u, v), 0.1 + 0.15 * k)
            rad_map[(u, v)] = rad
            edge_color = edge_colors[i]
            nx.draw_networkx_edges(
                G_vis, pos, edgelist=[(u, v)],
                connectionstyle=f'arc3, rad={rad}',
                edge_color=edge_color,
                arrows=True, arrowstyle='-|>', arrowsize=10, width=1 if edge_color == 'red' else 0.5
            )
        # 노드 그리기
        nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=60, alpha=0.9)

        # 범례 추가
        legend_elements = [
            Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='User_Card'),
            Patch(facecolor='orange', edgecolor='orange', label='Merchant'),
            Patch(facecolor='gray', edgecolor='gray', label='Other'),
            Patch(facecolor='red', edgecolor='red', label='Fraudulent Transaction', linewidth=2),
            Patch(facecolor='lightgray', edgecolor='lightgray', label='Normal Transaction', linewidth=1)
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('Transaction Graph Visualization')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_edges(self, start_date=None, end_date=None):
        if self.edges is None:
            raise ValueError("Edges dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")
        if start_date is not None and end_date is not None:
            mask = (self.edges['Date'] >= pd.to_datetime(start_date)) & (self.edges['Date'] < pd.to_datetime(end_date))
            return self.edges.loc[mask].reset_index(drop=True)
        elif start_date is not None and end_date is None:
            mask = (self.edges['Date'] >= pd.to_datetime(start_date))
            return self.edges.loc[mask].reset_index(drop=True)
        elif start_date is None and end_date is not None:
            mask = (self.edges['Date'] < pd.to_datetime(end_date))
            return self.edges.loc[mask].reset_index(drop=True)
        else:
            return self.edges

    def get_node_to_idx(self):
        if not self.node_to_idx:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        return self.node_to_idx
    
    def get_idx_to_node(self):
        if not self.idx_to_node:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        return self.idx_to_node