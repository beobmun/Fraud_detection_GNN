import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, HeteroData
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from tqdm import tqdm

class IBM_Dataset:
    def __init__(self):
        self.transactions_df = None
        self.users_df = None
        self.cards_df = None

        self.edge_transactions = None

        self.node_users = None
        self.node_cards = None
        self.node_merchants = None

        self.node_to_idx = dict()
        self.idx_to_node = dict()
        # self.user_to_idx = dict()
        # self.idx_to_user = dict()
        self.card_to_idx = dict()
        self.idx_to_card = dict()
        self.merchant_to_idx = dict()
        self.idx_to_merchant = dict()
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
        
        with tqdm(total=9 ,desc="Preprocessing transactions...", ncols=100, leave=False) as pbar:
            self.edge_transactions = self.transactions_df.copy()
            # Create a 'DateTime' column
            self.edge_transactions['DateTime'] = pd.to_datetime(
                self.edge_transactions['Year'].astype(str) + '-' +
                self.edge_transactions['Month'].astype(str).str.zfill(2) + '-' +
                self.edge_transactions['Day'].astype(str).str.zfill(2) + ' ' +
                self.edge_transactions['Time']
            )
            pbar.update(1)
            
            # Create a separate 'Date' column
            self.edge_transactions['Date'] = pd.to_datetime(
                self.edge_transactions['Year'].astype(str) + '-' +
                self.edge_transactions['Month'].astype(str).str.zfill(2) + '-' +
                self.edge_transactions['Day'].astype(str).str.zfill(2)
            )
            self.edge_transactions = self.edge_transactions.sort_values(by='DateTime', ascending=True)
            self.edge_transactions = self.edge_transactions.reset_index(drop=True)
            
            self.edge_transactions['isFraud'] = self.edge_transactions['Is Fraud?'].map({'No': 0, 'Yes': 1})
            pbar.update(1)
            
            # Select relevant columns
            self.edge_transactions = self.edge_transactions[['DateTime', 'Date', 'User', 'Card', 'Merchant Name', 'Amount', 'Use Chip', 'Zip', 'MCC', 'Errors?', 'isFraud']]

            self.edge_transactions = self.edge_transactions.rename(columns={'Errors?': 'Error'})
            self.edge_transactions['User_Card'] = self.edge_transactions['User'].astype(str) + '_' + self.edge_transactions['Card'].astype(str)

            self.edge_transactions.loc[:, 'Amount'] = self.edge_transactions['Amount'].replace({"\$": "", ",": ""}, regex=True).astype(float)
            self.edge_transactions['Src'] = np.where(self.edge_transactions['Amount'] >= 0, self.edge_transactions['User_Card'], self.edge_transactions['Merchant Name']).astype(str)
            self.edge_transactions['Dest'] = np.where(self.edge_transactions['Amount'] >= 0, self.edge_transactions['Merchant Name'], self.edge_transactions['User_Card']).astype(str)
            self.edge_transactions['Relation'] = np.where(self.edge_transactions['Amount'] >= 0, 'transaction', 'refund')
            pbar.update(1)
            
            # Amount scaling
            scaler = MinMaxScaler()
            abs_amounts = self.edge_transactions[['Amount']].abs().astype(float)
            log_scaled_amounts = np.log1p(abs_amounts)
            self.edge_transactions['Scaled_Amount'] = scaler.fit_transform(log_scaled_amounts)
            pbar.update(1)

            # MCC indexing
            all_mcc = self.edge_transactions['MCC'].unique()
            self.mcc_to_idx = {mcc: idx for idx, mcc in enumerate(all_mcc)}
            self.idx_to_mcc = {idx: mcc for mcc, idx in self.mcc_to_idx.items()}
            self.edge_transactions['MCC_idx'] = self.edge_transactions['MCC'].map(self.mcc_to_idx)
            pbar.update(1)

            # Zip code indexing
            self.edge_transactions['Zip'] = self.edge_transactions['Zip'].fillna(0).astype(float)
            all_zips = self.edge_transactions['Zip'].unique()
            self._set_zip_mappings(all_zips)
            self.edge_transactions['Zip_idx'] = self.edge_transactions['Zip'].map(self.zip_to_idx)
            pbar.update(1)

            # Use Chip, Error to one-hot encoding
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            use_chip_onehot = onehot_encoder.fit_transform(self.edge_transactions[['Use Chip']])
            use_chip_types = onehot_encoder.get_feature_names_out(['Use Chip'])
            use_chip_df = pd.DataFrame(use_chip_onehot, columns=use_chip_types, index=self.edge_transactions.index)

            self.edge_transactions['Error'] = self.edge_transactions['Error'].fillna('NaN')
            errors = self.edge_transactions['Error'].unique()
            error_types = set()
            for errs in errors:
                if errs == 'NaN':
                    continue
                for err in errs.split(','):
                    error_types.add("Error_" + err.strip())
            error_df = pd.DataFrame(0, columns=list(error_types), index=self.edge_transactions.index, dtype=float)
            for i, errs in enumerate(self.edge_transactions['Error']):
                if errs == 'NaN':
                    continue
                for err in errs.split(','):
                    error_df.at[i, "Error_" + err.strip()] = 1

            self.edge_transactions = pd.concat([self.edge_transactions, use_chip_df, error_df], axis=1)
            pbar.update(1)

            # Node Merchants
            merchant_names = self.edge_transactions['Merchant Name'].unique()
            merchant_zips = self.edge_transactions.groupby('Merchant Name')['Zip_idx'].first().to_dict()
            self.node_merchants = pd.DataFrame({
                'Merchant Name': merchant_names,
                'Zip_idx': [merchant_zips[name] for name in merchant_names]
            })
            self.node_merchants['Merchant Name'] = self.node_merchants['Merchant Name'].astype(str)
            pbar.update(1)

            # Drop intermediate columns
            self.edge_transactions = self.edge_transactions.drop(columns=['DateTime', 'User', 'Card', 'Merchant Name', 'Amount', 'Use Chip', 'Zip', 'MCC', 'Error', 'User_Card'])
            pbar.update(1)

        print("Preprocessing transactions completed.")

        return self
    
    def preprocess_users(self):
        if self.users_df is None:
            raise ValueError("Users dataframe is not loaded. Please call read_users_csv() first.")
        
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

        # One-hot encoding for FICO_Category
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        fico_onehot = onehot_encoder.fit_transform(self.users_df[['FICO_Category']])
        fico_types = onehot_encoder.get_feature_names_out(['FICO_Category'])
        fico_df = pd.DataFrame(fico_onehot, columns=fico_types, index=self.users_df.index)
        self.node_users = pd.concat([self.users_df, fico_df], axis=1)
        self.node_users = self.node_users.drop(columns=['FICO_Category'])

        print("Preprocessing users completed.")

        return self

    def preprocess_cards(self):
        if self.cards_df is None:
            raise ValueError("Cards dataframe is not loaded. Please call read_cards_csv() first.")
        
        # Create a 'User_Card' column
        self.cards_df['User_Card'] = self.cards_df['User'].astype(str) + "_" + self.cards_df['CARD INDEX'].astype(str)
        # Select relevant columns
        self.cards_df = self.cards_df[['User', 'User_Card', 'Card Brand', 'Card Type']]

        self.node_cards = self.cards_df.copy()

        # One-hot encoding for categorical features
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        card_brand_onehot = onehot_encoder.fit_transform(self.node_cards[['Card Brand']])
        card_brand_types = onehot_encoder.get_feature_names_out(['Card Brand'])
        card_brand_df = pd.DataFrame(card_brand_onehot, columns=card_brand_types, index=self.node_cards.index)
        card_type_onehot = onehot_encoder.fit_transform(self.node_cards[['Card Type']])
        card_type_types = onehot_encoder.get_feature_names_out(['Card Type'])
        card_type_df = pd.DataFrame(card_type_onehot, columns=card_type_types, index=self.node_cards.index)
        self.node_cards = pd.concat([self.node_cards, card_brand_df, card_type_df], axis=1)

        self.node_cards = self.node_cards.drop(columns=['Card Brand', 'Card Type'])

        print("Preprocessing cards completed.")

        return self
        
    def _set_zip_mappings(self, all_zips):
        for zip in all_zips:
            if zip not in self.zip_to_idx:
                idx = len(self.zip_to_idx)
                self.zip_to_idx[zip] = idx
                self.idx_to_zip[idx] = zip

    def create_node_mappings(self):
        if self.edge_transactions is None:
            raise ValueError("Edge transactions dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")
        if self.node_merchants is None:
            raise ValueError("Node merchants dataframe is not created. Please call preprocess_transactions() first.")
        if self.node_cards is None:
            raise ValueError("Node cards dataframe is not created. Please call preprocess_user_cards() first.")

        # User_Card indexing
        user_cards = self.node_cards['User_Card'].unique()
        self.card_to_idx = {card: idx for idx, card in enumerate(user_cards)}
        self.idx_to_card = {idx: card for card, idx in self.card_to_idx.items()}

        # Merchant indexing
        merchants = self.node_merchants['Merchant Name'].unique()
        self.merchant_to_idx = {merchant: idx for idx, merchant in enumerate(merchants)}
        self.idx_to_merchant = {idx: merchant for merchant, idx in self.merchant_to_idx.items()}

        # Combine all unique nodes from edges
        src_nodes = self.edge_transactions['Src'].unique()
        dest_nodes = self.edge_transactions['Dest'].unique()
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

    '''
    def build_pyg_graph(self, start_date=None, end_date=None):
        if self.edge_transactions is None:
            raise ValueError("Edge transactions dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")
        if not self.node_to_idx:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        if self.x is None:
            raise ValueError("Node features are not created. Please call create_graph_x() first.")
        
        filtered_edges = self.get_edges(start_date, end_date)
        src_idx = filtered_edges['Src'].map(self.node_to_idx).to_numpy()
        dest_idx = filtered_edges['Dest'].map(self.node_to_idx).to_numpy()
        edge_index = torch.tensor(np.vstack((src_idx, dest_idx)), dtype=torch.long)

        drop_cols = {'Date', 'Src', 'Dest', 'isFraud', 'MCC_idx', 'Zip_idx', 'Relation'}
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
    '''

    def build_hetero_graph(self, start_date=None, end_date=None):
        if self.node_cards is None:
            raise ValueError("Node cards dataframe is not created. Please call preprocess_user_cards() first.")
        if self.node_merchants is None:
            raise ValueError("Node merchants dataframe is not created. Please call preprocess_transactions() first.")

        data = HeteroData()
        card_attr = self.node_cards.drop(columns=['User', 'User_Card']).values
        user_attr = self.node_users.drop(columns=['User', 'Zip_idx']).values
        merchant_attr = self.node_merchants.drop(columns=['Merchant Name', 'Zip_idx']).values

        data['card'].x = torch.tensor(card_attr, dtype=torch.float)
        data['card'].num_nodes = data['card'].x.size(0)

        data['user'].x = torch.tensor(user_attr, dtype=torch.float)
        data['user'].zip_idx = torch.tensor(self.node_users['Zip_idx'].values, dtype=torch.long)
        data['user'].num_nodes = data['user'].x.size(0)

        data['merchant'].x = torch.tensor(merchant_attr, dtype=torch.float)
        data['merchant'].zip_idx = torch.tensor(self.node_merchants['Zip_idx'].values, dtype=torch.long)
        data['merchant'].num_nodes = data['merchant'].x.size(0)

        filtered_edge_transactions = self.get_edge_transactions(start_date, end_date)

        # data['card', 'transaction', 'merchant']
        transactions = filtered_edge_transactions[filtered_edge_transactions['Relation'] == 'transaction']
        src_idx = transactions['Src'].map(self.card_to_idx).to_numpy()
        dest_idx = transactions['Dest'].map(self.merchant_to_idx).to_numpy()
        data['card', 'transaction', 'merchant'].edge_index = torch.tensor(np.vstack((src_idx, dest_idx)), dtype=torch.long)
        transactions_attr_cols = ['Scaled_Amount']
        transactions_attr_cols += [col for col in filtered_edge_transactions.columns if col.startswith('Use Chip_')]
        transactions_attr_cols += [col for col in filtered_edge_transactions.columns if col.startswith('Error_')]
        data['card', 'transaction', 'merchant'].edge_attr = torch.tensor(transactions[transactions_attr_cols].values, dtype=torch.float)
        data['card', 'transaction', 'merchant'].edge_labels = torch.tensor(transactions['isFraud'].values, dtype=torch.long)
        data['card', 'transaction', 'merchant'].edge_mcc_idx = torch.tensor(transactions['MCC_idx'].values, dtype=torch.long)
        data['card', 'transaction', 'merchant'].edge_zip_idx = torch.tensor(transactions['Zip_idx'].values, dtype=torch.long)
        # data['merchant', 'transaction_by', 'card']
        data['merchant', 'transaction_by', 'card'].edge_index = torch.tensor(np.vstack((dest_idx, src_idx)), dtype=torch.long)
        data['merchant', 'transaction_by', 'card'].edge_attr = torch.tensor(transactions[transactions_attr_cols].values, dtype=torch.float)
        data['merchant', 'transaction_by', 'card'].edge_mcc_idx = torch.tensor(transactions['MCC_idx'].values, dtype=torch.long)
        data['merchant', 'transaction_by', 'card'].edge_zip_idx = torch.tensor(transactions['Zip_idx'].values, dtype=torch.long)

        # data['merchant', 'refund', 'card']
        refunds = filtered_edge_transactions[filtered_edge_transactions['Relation'] == 'refund']
        src_idx = refunds['Src'].map(self.merchant_to_idx).to_numpy()
        dest_idx = refunds['Dest'].map(self.card_to_idx).to_numpy()
        data['merchant', 'refund', 'card'].edge_index = torch.tensor(np.vstack((src_idx, dest_idx)), dtype=torch.long)
        data['merchant', 'refund', 'card'].edge_attr = torch.tensor(refunds[transactions_attr_cols].values, dtype=torch.float)
        data['merchant', 'refund', 'card'].edge_labels = torch.tensor(refunds['isFraud'].values, dtype=torch.long)
        data['merchant', 'refund', 'card'].edge_mcc_idx = torch.tensor(refunds['MCC_idx'].values, dtype=torch.long)
        data['merchant', 'refund', 'card'].edge_zip_idx = torch.tensor(refunds['Zip_idx'].values, dtype=torch.long)
        # data['card', 'refund_by', 'merchant']
        data['card', 'refund_by', 'merchant'].edge_index = torch.tensor(np.vstack((dest_idx, src_idx)), dtype=torch.long)
        data['card', 'refund_by', 'merchant'].edge_attr = torch.tensor(refunds[transactions_attr_cols].values, dtype=torch.float)
        data['card', 'refund_by', 'merchant'].edge_mcc_idx = torch.tensor(refunds['MCC_idx'].values, dtype=torch.long)
        data['card', 'refund_by', 'merchant'].edge_zip_idx = torch.tensor(refunds['Zip_idx'].values, dtype=torch.long)
        
        # data['card', 'same_merchant', 'card']
        # merchant_grouped_transactions = transactions.groupby('Dest')['Src'].apply(set)
        # edge_same_merchants = []
        # for merchant, cards_group in merchant_grouped_transactions.items():
        #     if len(cards_group) < 2:
        #         continue
        #     cards_list = list(cards_group)
        #     for i in range(len(cards_list)):
        #         for j in range(i + 1, len(cards_list)):
        #             card_a = cards_list[i]
        #             card_b = cards_list[j]
        #             card_a_idx = self.card_to_idx[card_a]
        #             card_b_idx = self.card_to_idx[card_b]
        #             edge_same_merchants.append((card_a_idx, card_b_idx))
        #             edge_same_merchants.append((card_b_idx, card_a_idx))
        # edge_same_merchants = torch.tensor(edge_same_merchants, dtype=torch.long).t()
        # data['card', 'same_merchant', 'card'].edge_index = edge_same_merchants

        # data['card', 'belong_to', 'user'], data['user', 'own', 'card']
        belong_to = self.node_cards.groupby('User')['User_Card'].apply(list)
        edge_belong_to = []
        edge_owns = []
        for user_idx, cards in belong_to.items():
            for card in cards:
                card_idx = self.card_to_idx[card]
                edge_belong_to.append((card_idx, user_idx))
                edge_owns.append((user_idx, card_idx))
        edge_belong_to = torch.tensor(edge_belong_to, dtype=torch.long).t()
        edge_owns = torch.tensor(edge_owns, dtype=torch.long).t()
        data['card', 'belong_to', 'user'].edge_index = edge_belong_to
        data['user', 'own', 'card'].edge_index = edge_owns

        return data

    def show_hetero_graph(self, data):
        # --- 1. 현재 스냅샷 거래에 사용되는 노드만 추출 ---
        # transaction과 refund 엣지에 등장하는 card 및 merchant 노드
        trans_card_nodes_in_snap = set(data['card', 'transaction', 'merchant'].edge_index[0].tolist())
        trans_merchant_nodes_in_snap = set(data['card', 'transaction', 'merchant'].edge_index[1].tolist())

        refund_card_nodes_in_snap = set(data['merchant', 'refund', 'card'].edge_index[1].tolist())
        refund_merchant_nodes_in_snap = set(data['merchant', 'refund', 'card'].edge_index[0].tolist())

        # 스냅샷에 나타나는 모든 card, merchant 노드
        used_card_nodes_in_snap = trans_card_nodes_in_snap.union(refund_card_nodes_in_snap)
        used_merchant_nodes_in_snap = trans_merchant_nodes_in_snap.union(refund_merchant_nodes_in_snap)

        # 'own' 엣지로 연결되어 있고, 스냅샷에 사용된 card와 연결된 user 노드만 추출
        # data['user', 'own', 'card'].edge_index[0]은 user_idx, [1]은 card_idx
        used_user_nodes_in_snap = set()
        own_ei = data['user', 'own', 'card'].edge_index
        for i in range(own_ei.shape[1]):
            user_idx = own_ei[0, i].item() # .item()으로 텐서에서 숫자 추출
            card_idx = own_ei[1, i].item()
            if card_idx in used_card_nodes_in_snap:
                used_user_nodes_in_snap.add(user_idx)

        # 'co_transactions_at' 엣지에 등장하는 card 노드
        co_trans_card_nodes_in_snap = set(data['card', 'same_merchant', 'card'].edge_index[0].tolist()).union(
                                        set(data['card', 'same_merchant', 'card'].edge_index[1].tolist()))
        used_card_nodes_in_snap.update(co_trans_card_nodes_in_snap) # co_trans 엣지 때문에 사용되는 card도 포함


        # 2. NetworkX MultiDiGraph 생성
        G = nx.MultiDiGraph()

        # 노드 추가 (타입별로, 스냅샷에 사용되는 노드만)
        for idx in used_card_nodes_in_snap:
            G.add_node(f'card_{idx}', ntype='card', idx=idx)
        for idx in used_merchant_nodes_in_snap:
            G.add_node(f'merchant_{idx}', ntype='merchant', idx=idx)
        for idx in used_user_nodes_in_snap:
            G.add_node(f'user_{idx}', ntype='user', idx=idx)

        # 3. transaction edge 추가 (card→merchant, 곡선, fraud는 빨간색)
        trans_ei = data['card', 'transaction', 'merchant'].edge_index
        trans_labels = data['card', 'transaction', 'merchant'].edge_labels.numpy()
        for i in range(trans_ei.shape[1]):
            src_pyg_idx = trans_ei[0, i].item()
            dst_pyg_idx = trans_ei[1, i].item()
            src = f'card_{src_pyg_idx}'
            dst = f'merchant_{dst_pyg_idx}'
            
            # 해당 노드가 그래프에 추가된 경우에만 엣지 추가
            if src in G.nodes and dst in G.nodes:
                fraud = trans_labels[i] == 1
                G.add_edge(src, dst, etype='transaction', fraud=fraud, key=f'trans_{i}')

        # 4. refund edge 추가 (merchant→card, 곡선, fraud는 빨간색)
        refund_ei = data['merchant', 'refund', 'card'].edge_index
        refund_labels = data['merchant', 'refund', 'card'].edge_labels.numpy()
        for i in range(refund_ei.shape[1]):
            src_pyg_idx = refund_ei[0, i].item()
            dst_pyg_idx = refund_ei[1, i].item()
            src = f'merchant_{src_pyg_idx}'
            dst = f'card_{dst_pyg_idx}'

            if src in G.nodes and dst in G.nodes:
                fraud = refund_labels[i] == 1
                G.add_edge(src, dst, etype='refund', fraud=fraud, key=f'refund_{i}')

        # 5. (user, own, card) edge 추가 (user→card, 화살표 없음, 직선, 노란색 계열)
        own_ei = data['user', 'own', 'card'].edge_index
        for i in range(own_ei.shape[1]):
            src_pyg_idx = own_ei[0, i].item()
            dst_pyg_idx = own_ei[1, i].item()
            src = f'user_{src_pyg_idx}'
            dst = f'card_{dst_pyg_idx}'
            
            # 스냅샷에 사용되는 user_card 노드와 연결된 user 노드만 포함
            if src in G.nodes and dst in G.nodes: # G.nodes에 이미 추가된 노드만 확인
                G.add_edge(src, dst, etype='own', fraud=False, key=f'own_{i}', arrow=False) # arrow=False는 draw_networkx_edges에서 다시 처리

        # 6. ('card', 'same_merchant', 'card') 엣지 추가 (직선, 검은색 계열)
        co_trans_ei = data['card', 'same_merchant', 'card'].edge_index
        # co_trans 엣지에 대한 레이블이 없다고 가정 (있다면 동일하게 fraud=True/False 설정 가능)
        for i in range(co_trans_ei.shape[1]):
            src_pyg_idx = co_trans_ei[0, i].item()
            dst_pyg_idx = co_trans_ei[1, i].item()
            src = f'card_{src_pyg_idx}'
            dst = f'card_{dst_pyg_idx}'

            if src in G.nodes and dst in G.nodes:
                # same_merchant 엣지는 기본적으로 fraud=False로 설정 (레이블이 없다면)
                G.add_edge(src, dst, etype='same_merchant', fraud=False, key=f'co_trans_{i}', arrow=False)


        # 7. node 색상 매핑
        node_colors = []
        node_color_map = {'card': 'dodgerblue', 'merchant': 'orange', 'user': 'limegreen'}
        for n, attr in G.nodes(data=True):
            node_colors.append(node_color_map.get(attr['ntype'], 'gray'))

        # 8. node 위치 (spring layout)
        pos = nx.spring_layout(G, seed=42, k=0.8, iterations=50) # k와 iterations 조정하여 노드 간 간격 및 안정화 개선

        plt.figure(figsize=(16, 12))

        # 9. 노드 그리기
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.9) # 노드 크기 약간 키움

        # 10. edge 그리기
        rad_map = {} # 같은 (u,v)쌍에 여러 엣지가 있을 때 곡률을 주기 위함

        for u, v, k, attr in G.edges(keys=True, data=True):
            edge_type = attr['etype']
            is_fraud = attr.get('fraud', False) # fraud 속성이 없을 경우 기본값 False

            if edge_type in ['transaction', 'refund']:
                color = 'red' if is_fraud else 'lightgray'
                width = 2.5 if is_fraud else 1.5 # 사기 엣지 두껍게
                
                # 같은 (u,v) 쌍에 대해 다른 곡률 (MultiDiGraph를 다룰 때 유용)
                # 이 부분이 MultiDiGraph 시각화의 핵심
                # G[u][v]는 u에서 v로 가는 모든 엣지 딕셔너리
                # list(G[u][v]).index(k)로 현재 엣지(k)가 해당 쌍에서 몇 번째인지 확인하여 곡률 조절
                arc_rad = 0.1 + 0.15 * list(G[u][v]).index(k) if len(G[u][v]) > 1 else 0.1
                
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    edge_color=color,
                    connectionstyle=f'arc3,rad={arc_rad}', # 곡선
                    arrows=True, arrowstyle='-|>', arrowsize=14, width=width, alpha=0.9
                )
            elif edge_type == 'own':
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    edge_color='gold', # 노란색 계열
                    arrows=False, width=1.5, style='solid', alpha=0.7 # 화살표 없음, 두께 조절
                )
            elif edge_type == 'same_merchant': # 'card', 'same_merchant', 'card'
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    edge_color='dimgray', # 어두운 회색 계열
                    arrows=False, width=1, style='dotted', alpha=0.6 # 화살표 없음, 점선
                )

        # 11. 노드 라벨 (선택 사항, 노드가 많으면 지저분해질 수 있음)
        # nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # 12. 범례
        legend_elements = [
            Patch(facecolor='dodgerblue', edgecolor='k', label='Card'),
            Patch(facecolor='orange', edgecolor='k', label='Merchant'),
            Patch(facecolor='limegreen', edgecolor='k', label='User'),
            Line2D([0], [0], color='red', lw=2.5, label='Fraud Transaction/Refund'),
            Line2D([0], [0], color='lightgray', lw=1.5, label='Normal Transaction/Refund'),
            Line2D([0], [0], color='gold', lw=1.5, label='Own Edge (User↔Card)', linestyle='solid'),
            Line2D([0], [0], color='dimgray', lw=1, label='Same Merchant Edge (Card↔Card)', linestyle='dotted')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title(f"HeteroData Graph Visualization)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    '''
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
    '''

    def get_edge_transactions(self, start_date=None, end_date=None):
        if self.edge_transactions is None:
            raise ValueError("Edge transactions dataframe is not loaded. Please call read_transactions_csv() and preprocess_transactions() first.")
        if start_date is not None and end_date is not None:
            if pd.to_datetime(start_date) == pd.to_datetime(end_date):
                mask = (self.edge_transactions['Date'] == pd.to_datetime(start_date))
                return self.edge_transactions.loc[mask].reset_index(drop=True)
            else:
                mask = (self.edge_transactions['Date'] >= pd.to_datetime(start_date)) & (self.edge_transactions['Date'] < pd.to_datetime(end_date))
                return self.edge_transactions.loc[mask].reset_index(drop=True)
        elif start_date is not None and end_date is None:
            mask = (self.edge_transactions['Date'] >= pd.to_datetime(start_date))
            return self.edge_transactions.loc[mask].reset_index(drop=True)
        elif start_date is None and end_date is not None:
            mask = (self.edge_transactions['Date'] < pd.to_datetime(end_date))
            return self.edge_transactions.loc[mask].reset_index(drop=True)
        else:
            return self.edge_transactions

    def get_node_to_idx(self):
        if not self.node_to_idx:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        return self.node_to_idx
    
    def get_idx_to_node(self):
        if not self.idx_to_node:
            raise ValueError("Node mappings are not created. Please call create_node_mappings() first.")
        return self.idx_to_node