import torch
from IBM_dataset import IBM_Dataset
from dataloader import Dataset, collate_fn
from model import FraudDetectionModel
from train import Train
from setproctitle import setproctitle

TRANSACTIONS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/credit_card_transactions-ibm_v2.csv'
USERS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/sd254_users.csv'
CARDS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/sd254_cards.csv'

METADATA = (
    ['card', 'merchant', 'user'],
    [
        ('card', 'transaction', 'merchant'),
        ('merchant', 'refund', 'card'),
        ('card', 'same_merchant', 'card'),
        ('card', 'belong_to', 'user'),
        ('user', 'own', 'card'),
    ]
)

NODE_FEATURES_DIM = {
    'card': 4 + 3, # Card Brand (4), Card Type (3) // data['card'].x.shape[1]
    'merchant': 0, # Merchant에는 현재 노드 특성 없음  // data['merchant'].x.shape[1]
    'user': 5 # FICO_Category (5) // data['user'].x.shape[1]
}

EDGE_FEATURES_DIM = {
    ('card', 'transaction', 'merchant'): 1 + 3 + 7, # Scaled_Amount (1), Use Chip (3), Error (7) // data['card', 'transaction', 'merchant'].edge_attr.shape[1]
    ('merchant', 'refund', 'card'): 1 + 3 + 7,
    ('card', 'same_merchant', 'card'): 0, # same_merchant 엣지에는 현재 특성 없음
    ('card', 'belong_to', 'user'): 0, # belong_to 엣지에는 현재 특성 없음
    ('user', 'own', 'card'): 0 # own 엣지에는 현재 특성 없음
}

ZIP_EMB_DIM = 64
MCC_EMB_DIM = 64
GNN_HIDDEN_DIM = 128
GRU_HIDDEN_DIM = 128
ATTENTION_HEADS = 4

# NUM_ZIP_IDX = len(dataset.zip_to_idx)
# NUM_MCC_IDX = len(dataset.mcc_to_idx)

IN_CHANNELS_DICT = {**NODE_FEATURES_DIM, **EDGE_FEATURES_DIM}
OUT_CHANNELS = 2  # 이진 분류 (사기/정상)

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        setproctitle("beobmun_GNN_training")

        dataset = (IBM_Dataset()
                .read_transactions_csv(TRANSACTIONS_CSV_PATH)
                .read_users_csv(USERS_CSV_PATH)
                .read_cards_csv(CARDS_CSV_PATH)
                .preprocess_transactions()
                .preprocess_users()
                .preprocess_cards()
                .create_node_mappings()
                )
        
        model = FraudDetectionModel(
            NODE_FEATURES_DIM, EDGE_FEATURES_DIM, ZIP_EMB_DIM, MCC_EMB_DIM, GNN_HIDDEN_DIM, GRU_HIDDEN_DIM, METADATA, 
            len(dataset.zip_to_idx), len(dataset.mcc_to_idx), 
            ATTENTION_HEADS)
        model.to(device)

        epochs = 100
        learning_rate = 0.001
        batch_size=2
        window_size=1
        memory_size=10

        start_date = '1996-01-01'
        end_date = '2019-12-31'
        # end_date = '1996-02-28'

        train = (Train()
                 .set_dataset(dataset)
                 .set_model(model)
                 .set_device(device)
                 .set_dataloaders(start_date, end_date, window_size=window_size, memory_size=memory_size, batch_size=batch_size)
                 )

        train.run_training(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # for fold, (train_key, val_key) in enumerate(zip(train.train_dataloaders.keys(), train.val_dataloaders.keys())):
        #     print(f"Starting training for fold {fold + 1} | Train period: {train_key}, Val period: {val_key}")
        #     train_dataloader = train.train_dataloaders[train_key]
        #     val_dataloader = train.val_dataloaders[val_key]
        #     train.run_training(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        #                        train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        #     print(f"Completed training for fold {fold + 1}")

    except Exception as e:
        print(f"Error occurred: {e}")
        return
    


if __name__ == "__main__":
    main()