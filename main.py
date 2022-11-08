import os
import copy
import json
import logging
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from architectures.TSMC import TSMC
from architectures.classifier import DenseClassifier
from datasets.UCIHAR_dataset import UCIHARDataModule
from datasets.cho2017_dataset import Cho2017DataModule
from modules.encoding_module import plEncodingModule
from modules.classification_module import plClassificationModule
from datasets.seed_dataset import SEEDDataModule
from utils.restricted_float import restricted_float
#from knockknock import email_sender
#from dotenv import load_dotenv

#load_dotenv()
#@email_sender(recipient_emails=[os.environ.get("recipient_emails")], sender_email=os.environ.get("sender_email"))
def main(args):
    logging.getLogger("lightning").setLevel(logging.WARNING)
    pl.seed_everything(33)

    ### CONFIG AND HYPERPARAMETERS
    with open("device_hyperparameters.json") as f:
        device_params = json.load(f)
    log_dir = device_params['log_dir']
    num_workers = device_params['num_workers']
    batch_size = device_params['ss_batch_size']
    limit_train_batches = device_params['limit_train_batches']
    limit_val_batches = device_params['limit_val_batches']
    limit_test_batches = device_params['limit_test_batches']

    if args.dataset=="SEED":
        run_name = "emotion_recognition"
        datamodule = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            args.preprocessing,
            "emotion",
            batch_size,
            num_workers
        )
    elif args.dataset=="UCIHAR":
        run_name = "activity_recognition"
        datamodule = UCIHARDataModule(
            device_params["ss_ucihar_datapath"],
            args.preprocessing,
            batch_size,
            num_workers
        )
    elif args.dataset=="SEEDUC":
        run_name = "user_recognition"
        datamodule = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            args.preprocessing,
            "userID",
            batch_size,
            num_workers
        )
    elif args.dataset=="Cho2017":
        run_name = "motor_imagery"
        datamodule = Cho2017DataModule(
            device_params["ss_mi_datapath"],
            args.preprocessing,
            batch_size,
            num_workers
        )
    else:
        raise ValueError(f'parameter dataset has to be one of ["SEED", "UCIHAR", "SEEDUC", "Cho2017"], but got {args.dataset}')
    encoder = TSMC(
        input_features=datamodule.input_features,
        embedding_dim=args.embedding_dim,
        n_head_token_enc=args.n_head_token_enc,
        n_head_context_enc=args.n_head_context_enc,
        depth_context_enc=args.depth_context_enc,
        max_predict_len=args.max_predict_len
        )
    classifier = DenseClassifier(in_features=args.embedding_dim, out_features=datamodule.n_classes)

    encoder_module = plEncodingModule(
        encoder,
        batch_size,
        args.lr,
        args.tau,
        args.lam,
        args.masking_percentage,
        args.masking_method,
        num_workers
    )
    enc_classifier = plClassificationModule(
        copy.deepcopy(encoder),
        classifier,
        batch_size,
        args.lr,
        num_workers
    )
    pretrainer_checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=f"{log_dir}/checkpoints/{run_name}", save_last=True)
    pretrainer_csv_logger = CSVLogger(save_dir=f"{log_dir}/csv/", name=run_name)
    pretrainer = pl.Trainer(
        accelerator = "auto",
        default_root_dir=log_dir,
        max_epochs=args.pretrain_epochs,
        log_every_n_steps=1,
        callbacks=[
            pretrainer_checkpoint_callback,
        ],
        logger=[
            pretrainer_csv_logger,
            TensorBoardLogger(save_dir=f"{log_dir}/tb/", name=run_name, log_graph=False, default_hp_metric=False)
        ],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    supervised_trainer_checkpoint_callback = ModelCheckpoint(monitor="val_loss",dirpath=f"{log_dir}/checkpoints/{run_name}_classification", save_last=True)
    supervised_trainer_csv_logger = CSVLogger(save_dir=f"{log_dir}/csv/", name=f"{run_name}_classification")
    supervised_trainer = pl.Trainer(
        accelerator = "auto",
        default_root_dir=log_dir,
        max_epochs=args.finetune_epochs,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=args.es_after_epochs),
            supervised_trainer_checkpoint_callback
        ],
        logger=[
            supervised_trainer_csv_logger,
            TensorBoardLogger(save_dir=f"{log_dir}/tb/", name=f"{run_name}_classification", log_graph=False, default_hp_metric=False)
        ],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    pretrainer_csv_logger.log_dir
    #### START OF PRETRAINING ####
    pretrainer.fit(encoder_module, datamodule)
    
    with open(os.path.join(pretrainer_csv_logger.log_dir,'best_model_path.txt'), 'w') as f:
        f.write(pretrainer_checkpoint_callback.best_model_path)
    lightning_checkpoint = torch.load(pretrainer_checkpoint_callback.best_model_path)
    encoder_module.load_state_dict(lightning_checkpoint["state_dict"])
    enc_classifier.encoder.load_state_dict(encoder_module.student.state_dict())

    supervised_trainer_csv_logger.log_dir
    #### START OF FINE-TUNING ####
    supervised_trainer.fit(enc_classifier, datamodule)
    
    with open(os.path.join(supervised_trainer_csv_logger.log_dir,'best_model_path.txt'), 'w') as f:
        f.write(supervised_trainer_checkpoint_callback.best_model_path)

    #results = supervised_trainer.test(enc_classifier, datamodule)
    #return results[0]['test_acc']

if __name__ == "__main__":
    from utils.dotdict import dotdict
    args = {
        "dataset": "Cho2017",
        "embedding_dim": 64,
        "n_head_token_enc": 2,
        "n_head_context_enc": 2,
        "depth_context_enc": 1,
        "max_predict_len": 6,
        "lr": 1e-4,
        "tau": 0.9,
        "lam": 1,
        "masking_percentage": 0.5,
        "masking_method": "temporal_window_masking",
        "pretrain_epochs": 1,
        "finetune_epochs": 1,
        "es_after_epochs": 1,
        "train_val_split": "random",
        "preprocessing": "standardize"
        }

    main(dotdict(args))