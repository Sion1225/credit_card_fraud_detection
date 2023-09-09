import optuna

storage = optuna.storages.RDBStorage("sqlite:///ensembledb/1.db")

study_name = "final_xgboost2"
study_id = storage.get_study_id_from_name(study_name)

storage.delete_study(study_id)