class hyperparams:

    #---------------handle labs and wav path-------------------#
    #-----only apply to solver-----#
    orimusic_txt = './data/ori_data/music.txt'
    orilabel_txt = './data/ori_data/labels.txt'
    orimp3_dir = './data/music/mp3'
    labs_vacab = './data/vocab_labels.txt'
    #-----only apply to solver-----#

    wavs_dir = './data/music/wav'
    music_info = './data/music.csv'
    label_info = './data/labels.csv'
    train_dir = './data/train_data' # store train tf_record file.
    eval_dir = './data/eval_data' # store eval tf_record file.
    test_dir = './data/test_data' # store test tf_record file.
    train_size = 5000
    eval_size = 500
    test_size = 500
    # must support train_size + eval_size + test_size <= data_size and shuffle_size >= train_size + eval_size + test_size
    shuffle_size = 200 # relate to train data size or eval data size or test data size.
    # data_size = 5000 # relate to train data size or eval data size or test data size.
    batch_size = 100 # relate to train data size or eval data size or test data size.
    num_epoches = 30 # relate to train data size or eval data size or test data size.
    per_steps = 100

    #--------------------handle wave---------------------------#
    prepro_mp = True # if pre process, do multiprocess or not
    sr = 22050
    hop_length = int(12.5 * 0.001 * sr)
    win_length = int(50 * 0.001 * sr)
    segment_length = 600
    ref_db = 20
    max_db = 100

    #-----------------train hyperparams------------------------#
    n_fft = 2048
    f_size = n_fft/2 + 1
    lab_size = 95
    lr = 0.001
    lr_decay_steps = 500
    lr_decay_rate = 0.1
    gpu_ids = [0] # if multi gpu, list len must larger than one
    log_dir = './log' # tensorboard store dir
    model_dir = './model' # checkpoint store dir
