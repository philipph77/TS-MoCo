from architectures.TSMC import TSMC
from modules.encoding_module import plEncodingModule

if __name__ == '__main__':
    encoder = TSMC(
        input_features=62,
        embedding_dim=50,
        n_head_token_enc=10,
        n_head_context_enc=10,
        depth_context_enc=4,
        max_predict_len=6
    )

    encoder_module = plEncodingModule(
        encoder,
        128,
        1e-4,
        0.9,
        1,
        0.5,
        "channel_wise",
        1
    )

    print(encoder_module)