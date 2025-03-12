class Pth_Profile:
    def __init__(self, pth_filename: str, patch_num_layers: int, char_num_layers: int, hidden_size: int, patch_len: int):
        self.pth_filename = pth_filename
        self.patch_num_layers = patch_num_layers
        self.char_num_layers = char_num_layers
        self.hidden_size = hidden_size
        self.patch_len = patch_len

# SMALL and MEDIUM doesn't work? It stucks when generating

SMALL = Pth_Profile(
    'weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth',
    12, 3, 768, 2048
)

MEDIUM = Pth_Profile(
    'weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth',
    16, 3, 1024, 2048
)

LARGE = Pth_Profile(
    'weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth',
    20, 6, 1280, 1024
)