class Setting:
    data_folder: str = '120_samples_database_cut/'
    train_data_len: int = 10000
    test_data_len: int = 1200
    start_epoch: int = 0
    stop_epoch: int = 1000
    train_batch_size: int = 16
    test_batch_size: int = 16
    train_aug: bool = True
    image_size: int = 84
    num_classes: int = 120
    save_freq: int = 5
    lr: float = 0.001
    alpha: float = 2.0
    checkpoint_dir: str = 'checkpoints/'
    resume: str = None
    es_scale: float = 0.1
    patience: int = 5