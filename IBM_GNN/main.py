from load_data import IBM_Dataset

TRANSACTIONS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/credit_card_transactions-ibm_v2.csv'
USERS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/sd254_users.csv'
CARDS_CSV_PATH = '../data/IBM_Credit_Card_Transaction/sd254_cards.csv'

def main():
    try:
        dataset = (IBM_Dataset()
                .read_transactions_csv(TRANSACTIONS_CSV_PATH)
                .read_users_csv(USERS_CSV_PATH)
                .read_cards_csv(CARDS_CSV_PATH)
                .preprocess_transactions()
                .preprocess_user_cards()
                .create_node_mappings()
                .create_graph_x()
                )
        
        print(dataset.get_edges().head())
        print(dataset.get_edges().tail())

        start_date = '2000-01-30'
        end_date = '2000-01-31'

        data = dataset.build_pyg_graph(start_date, end_date)
        print(data)

        dataset.show_graph(data)


    except Exception as e:
        print(f"Error occurred: {e}")
        return
    


if __name__ == "__main__":
    main()