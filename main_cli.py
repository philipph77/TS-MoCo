import argparse
from utils.restricted_float import restricted_float
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Multi-Source Version of the SSMDR Framework")
    
    parser.add_argument("embedding_dim", type=int, help="mumber of features in the latent representation")
    parser.add_argument("n_head_token_enc", type=int, help="number of heads of the token encoder")
    parser.add_argument("n_head_context_enc", type=int, help="number of heads of the context encoder")
    parser.add_argument("depth_context_enc", type=int, help="depth of the context encoder")
    parser.add_argument("max_predict_len", type=int, help="maximum future timesteps to predict")
    
    parser.add_argument("lr", type=float, help="used learning rate for optimization")
    parser.add_argument("tau", type=float, help="momentum value")
    parser.add_argument("lam", type=float, help="loss weight")
    parser.add_argument("masking_percentage", type=restricted_float, help="percentage of the input that shall be masked")
    parser.add_argument("masking_method", choices=["random", "channel_wise", "temporal"], help="witch masking algorithm shall be used")
    parser.add_argument("pretrain_epochs", type=int, help="number of epochs for self-supervised pretraining")
    parser.add_argument("finetune_epochs", type=int, help="number of epochs for supervised fine-tuning")
    parser.add_argument("es_after_epochs", type=int, help="Number of Epochs without improvement in validation loss to stop the training after")

    parser.add_argument("train_val_split",  choices=["random", "subject"], help="whether to split the validation-set randomly or to perform leave one subject out validation")
    args = parser.parse_args()
    main(args)