from hparams import hyperparams as hp

def get_FRR(y, y_hat, threshold):
    '''
    :param y: A 2d array. real labels. [N, hp.lab_size]
    :param y_hat: A 2d array. logit labels. [N, hp.lab_size]
    :param threshold: A float. value in (0, 1)
    :return: A float. False Reject Rate.
    '''
    reject = 0
    total = 0
    batch_size = y.shape[0]
    for i in range(batch_size):
        for j in range(hp.lab_size):
            if y_hat[i][j] >= threshold and y[i][j] == 1:
                total += 1
            elif y_hat[i][j] < threshold and y[i][j] == 1:
                total += 1
                reject += 1
    res = reject * 1.0 / total
    return res

def get_FAR(y, y_hat, threshold):
    '''
    :param y: A 2d array. real labels. [N, hp.lab_size]
    :param y_hat: A 2d array. logit labels. [N, hp.lab_size]
    :param threshold: A float. value in (0, 1)
    :return: A float. False Accept Rate.
    '''
    accept = 0
    total = 0
    batch_size = y.shape[0]
    for i in range(batch_size):
        for j in range(hp.lab_size):
            if y_hat[i][j] < threshold and y[i][j] == 1:
                total += 1
            elif y_hat[i][j] >= threshold and y[i][j] == 0:
                total += 1
                accept += 1
    res = accept * 1.0 / total
    return res

