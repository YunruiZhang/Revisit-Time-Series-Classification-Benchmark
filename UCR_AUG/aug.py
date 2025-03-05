import numpy as np

def add_seqpad(train_x, test_x, padding_length, std=0.01):
    train_x_aug = []
    test_x_aug = []
    
    for i in range(train_x.shape[0]):
        # for front pad
        front_pad = []
        cur_mean = train_x[i][0]
        for a in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            front_pad.append(cur_mean)
        front_pad = np.array(front_pad).flatten()
        front_pad = np.flip(front_pad)
        # for end pad
        end_pad = []
        cur_mean = train_x[i][-1]
        for b in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            end_pad.append(cur_mean)
            
        end_pad = np.array(end_pad).flatten()
        train_x_aug.append(np.concatenate([front_pad, train_x[i], end_pad]))
    
    
    for i in range(test_x.shape[0]):
        # for end pad
        front_pad = []
        cur_mean = test_x[i][0]
        for a in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            front_pad.append(cur_mean)
        front_pad = np.array(front_pad).flatten()
        front_pad = np.flip(front_pad)
        # for end pad
        end_pad = []
        cur_mean = test_x[i][-1]
        for b in range(padding_length):
            cur_mean = np.random.normal(cur_mean, std, 1)
            end_pad.append(cur_mean)
            
        end_pad = np.array(end_pad).flatten()
        test_x_aug.append(np.concatenate([front_pad, test_x[i], end_pad]))
        
        
    train_x_aug = np.array(train_x_aug)
    test_x_aug = np.array(test_x_aug)
    return train_x_aug, test_x_aug

def shift_ts(train_x, test_x, padding):
    padd = padding
    np.random.seed(0)
    
    train_x_aug_shifted = []
    test_x_aug_shifted = []

    for a in range(train_x.shape[0]):
        random_shift_left = np.random.randint(1, padd)  # Include max_value in the range
        random_shift_right = padd-random_shift_left
        train_x_aug_shifted.append(train_x[a][padd-random_shift_left:-(padd-random_shift_right)])

    for a in range(test_x.shape[0]):
        random_shift_left = np.random.randint(1, padd)  # Include max_value in the range
        random_shift_right = padd-random_shift_left
        test_x_aug_shifted.append(test_x[a][padd-random_shift_left:-(padd-random_shift_right)])

    train_x_aug_shifted=np.array(train_x_aug_shifted)
    test_x_aug_shifted=np.array(test_x_aug_shifted)
    
    return train_x_aug_shifted, test_x_aug_shifted