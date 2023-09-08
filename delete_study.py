import optuna

storage = optuna.storages.RDBStorage("sqlite:///nndb/1.db")

study_name = "NN_for_card_fraud_2"
study_id = storage.get_study_id_from_name(study_name)

storage.delete_study(study_id)