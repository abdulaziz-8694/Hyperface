class Config:
    batch_size = 32
    checkpoint_path = './hyperface-ckpt'
    model_save_path = './hyperface-model.h5'
    epochs = 5
    arch_type = 'alexNet'
    sample_data_file = 'data/hyf_data.npy'
    input_shape = (227, 227, 3)
    learning_rate = 0.00001