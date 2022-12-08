import argparse
from utils.restricted_float import restricted_float
from main_supervised import main_supervised

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Multi-Source Version of the SSMDR Framework")
    
    parser.add_argument("dataset", choices=["SEED", "SEEDIJCAI", "UCIHAR", "SEEDUC", "Cho2017", "DREAMER"],help="which dataset to use")
    parser.add_argument("pos_embeddings_alpha", type=float, help="mulitplication factor for the positional embedding vector")
    parser.add_argument("embedding_dim", type=int, help="mumber of features in the latent representation")
    parser.add_argument("n_head_token_enc", type=int, help="number of heads of the token encoder")
    parser.add_argument("n_head_context_enc", type=int, help="number of heads of the context encoder")
    parser.add_argument("depth_context_enc", type=int, help="depth of the context encoder")
    
    parser.add_argument("lr", type=float, help="used learning rate for optimization")
    parser.add_argument("finetune_epochs", type=int, help="number of epochs for supervised fine-tuning")
    parser.add_argument("es_after_epochs", type=int, help="Number of Epochs without improvement in validation loss to stop the training after")

    parser.add_argument("train_val_split",  choices=["random", "subject"], help="whether to split the validation-set randomly or to perform leave one subject out validation")
    parser.add_argument("preprocessing",  choices=["None", "standardize", "normalize"], help="what preprocessing to apply to the inputs, e.g. normalization or standardization")
    args = parser.parse_args()
    main_supervised(args)