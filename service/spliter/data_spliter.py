from sklearn.model_selection import train_test_split
import os


def data_spliter(df, label='14d', by='apply_risk_created_at', id='apply_risk_id', test_size=0.2, method='oot', random_state=7):
    """

    :param df: dataframe
    :param by:
    :param test_size:
    :param method:
    :param random_state:
    :return:
    """
    df = df.set_index(id)

    if method == 'oot':
        df = df.sort_values(by, ascending=False)
        del df[by]
        test = df.iloc[:int(df.shape[0] * test_size), ]
        train = df.iloc[int(df.shape[0] * test_size):, ]
    elif method == 'random':
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        raise Exception(f'method {method} is not defined.')

    train_y = train.pop(label)
    test_y = test.pop(label)

    # datapath = os.path.join(path, 'data_spliter')
    # if not os.path.exists(datapath):
    #     os.mkdir(datapath)
    #
    # train.to_pickle(os.path.join(datapath + 'train.pkl'))
    # train_y.to_pickle(os.path.join(datapath + 'train_y.pkl'))
    # test.to_pickle(os.path.join(datapath + 'test.pkl'))
    # test_y.to_pickle(os.path.join(datapath + 'test_y.pkl'))

    return train, train_y, test, test_y
