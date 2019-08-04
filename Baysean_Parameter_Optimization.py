import time

# REVIEW_FILE_CSV = 'reviews.csv'
SHUFFLED_REVIEW_FILE_CSV = 'shuffled.100000.reviews.csv'  # 'shuffled.reviews.csv'
import pandas as pd

# GLOBAL VAR df_all <-- all review data
start = time.time()
df_all = pd.read_csv(SHUFFLED_REVIEW_FILE_CSV)
csv_load_time = time.time() - start
print(str(round(csv_load_time, 2)) + ' seconds loading CSV into memory')
del pd, SHUFFLED_REVIEW_FILE_CSV, start, csv_load_time


def get_data(size, metric):
    return df_all[[metric, 'text']].head(size)


def data_prep(df, metric):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text.values)
    VOCAB_SIZE = len(tokenizer.word_index) + 1

    X = tokenizer.texts_to_sequences(df.text.values)
    X = pad_sequences(X)
    # Normalize Y to be between 0 and 1
    Y = df[metric] / max(df[metric])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    return X, Y, VOCAB_SIZE, X_train, X_test, Y_train, Y_test


def model_factory(X_train,
                  VOCAB_SIZE,
                  EMBED_OUTPUT_DIM,
                  RNN_TYPE,
                  RNN_LAYER_COUNT,
                  RNN_OUT,
                  RNN_DROPOUT,
                  USE_SPATIAL_DROPOUT,
                  SPATIAL_DROPOUT,
                  LEARNING_RATE):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, GRU, SpatialDropout1D

    model = Sequential()

    # TODO https://realpython.com/python-keras-text-classification/

    # https://keras.io/layers/embeddings/
    # keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
    model.add(Embedding(VOCAB_SIZE,
                        EMBED_OUTPUT_DIM,
                        mask_zero=True,
                        input_length=X_train.shape[1]))

    if USE_SPATIAL_DROPOUT:
        model.add(SpatialDropout1D(SPATIAL_DROPOUT))

    RNN_TYPE = int(round(RNN_TYPE))  # RNN_TYPE_DICT = {0:'gru',1:'lstm'}
    if (RNN_TYPE == 0):
        model.add(GRU(RNN_OUT, dropout=RNN_DROPOUT, recurrent_dropout=RNN_DROPOUT))
        if RNN_LAYER_COUNT > 1:
            for i in range(RNN_LAYER_COUNT):
                model.add(GRU(RNN_OUT, return_sequences=True, dropout=RNN_DROPOUT, recurrent_dropout=RNN_DROPOUT))
    else:
        if RNN_LAYER_COUNT > 1:
            for i in range(RNN_LAYER_COUNT):
                model.add(LSTM(RNN_OUT, return_sequences=True, dropout=RNN_DROPOUT, recurrent_dropout=RNN_DROPOUT))

        model.add(LSTM(RNN_OUT, dropout=RNN_DROPOUT, recurrent_dropout=RNN_DROPOUT))

    model.add(Dense(1, activation='linear'))
    adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae', 'accuracy'])
    return model


def evaluate_model(model, history, X_test, Y_test, BATCH_SIZE):
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'], '.-')
    plt.plot(history.history['val_acc'], '.-')
    plt.plot(history.history['loss'], '.-')
    plt.plot(history.history['val_loss'], '.-')

    plt.title('training')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='best')
    plt.show()

    loss, mae, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=BATCH_SIZE)
    print('loss(mse):' + str(loss))  # mse
    print('mae:' + str(mae))
    print('acc:' + str(acc))

    y_pred = model.predict(X_test)
    # y_pred[0:5]
    # Y_test[0:5]

    fig, ax = plt.subplots()
    ax.scatter(Y_test, y_pred)
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    return loss, mae, acc


def is_experiment_redundant(DATA_SIZE,
                            METRIC,
                            EMBED_OUTPUT_DIM,
                            RNN_TYPE,
                            RNN_LAYER_COUNT,
                            RNN_OUT,
                            RNN_DROPOUT,
                            RECURRENT_DROPOUT,
                            USE_SPATIAL_DROPOUT,
                            SPATIAL_DROPOUT,
                            EPOCH,
                            BATCH_SIZE,
                            LEARNING_RATE):
    import pandas as pd
    import numpy as np
    df = pd.read_csv('results.csv')
    q = df[
        (df.DATA_SIZE == DATA_SIZE) &
        (df.METRIC == METRIC) &
        (df.EMBED_OUTPUT_DIM == EMBED_OUTPUT_DIM) &
        (df.RNN_TYPE == RNN_TYPE) &
        (df.RNN_LAYER_COUNT == RNN_LAYER_COUNT) &
        (df.RNN_OUT == RNN_OUT) &
        (df.RNN_DROPOUT == RNN_DROPOUT) &
        (df.RECURRENT_DROPOUT == RECURRENT_DROPOUT) &
        (df.USE_SPATIAL_DROPOUT == USE_SPATIAL_DROPOUT) &
        (df.SPATIAL_DROPOUT == SPATIAL_DROPOUT) &
        (df.EPOCH == EPOCH) &
        (df.BATCH_SIZE == BATCH_SIZE) &
        (df.LEARNING_RATE == LEARNING_RATE)]

    if q.shape[0] > 0:
        result = np.mean(q.bayes_metric)
        return result
    else:
        return -999


def run_experiment(DATA_SIZE,
                   METRIC,
                   EMBED_OUTPUT_DIM,
                   RNN_TYPE,
                   RNN_LAYER_COUNT,
                   RNN_OUT,
                   RNN_DROPOUT,
                   RECURRENT_DROPOUT,
                   USE_SPATIAL_DROPOUT,
                   SPATIAL_DROPOUT,
                   EPOCH,
                   BATCH_SIZE,
                   LEARNING_RATE):
    import uuid
    import pandas as pd
    from datetime import datetime
    import time
    import numpy as np
    import json

    DATA_SIZE = int(round(DATA_SIZE))
    metric_dict = {0: 'stars', 1: 'funny', 2: 'useful', 3: 'cool'}
    METRIC = metric_dict[round(METRIC, 0)]
    EMBED_OUTPUT_DIM = int(round(EMBED_OUTPUT_DIM))
    RNN_LAYER_COUNT = int(round(RNN_LAYER_COUNT))
    RNN_OUT = int(round(RNN_OUT))
    USE_SPATIAL_DROPOUT = bool(int(round(USE_SPATIAL_DROPOUT)))
    EPOCH = int(round(EPOCH))
    BATCH_SIZE = int(round(BATCH_SIZE))

    RNN_DROPOUT = round(RNN_DROPOUT, 2)
    RECURRENT_DROPOUT = round(RECURRENT_DROPOUT, 2)
    SPATIAL_DROPOUT = round(SPATIAL_DROPOUT, 2)
    LEARNING_RATE = round(LEARNING_RATE, 2)

    test_redundancy = is_experiment_redundant(DATA_SIZE,
                                              METRIC,
                                              EMBED_OUTPUT_DIM,
                                              RNN_TYPE,
                                              RNN_LAYER_COUNT,
                                              RNN_OUT,
                                              RNN_DROPOUT,
                                              RECURRENT_DROPOUT,
                                              USE_SPATIAL_DROPOUT,
                                              SPATIAL_DROPOUT,
                                              EPOCH,
                                              BATCH_SIZE,
                                              LEARNING_RATE)

    if test_redundancy != -999:
        return test_redundancy

    df = get_data(DATA_SIZE, METRIC)
    X, Y, VOCAB_SIZE, X_train, X_test, Y_train, Y_test = data_prep(df, METRIC)

    model = model_factory(X_train,
                          VOCAB_SIZE,
                          EMBED_OUTPUT_DIM,
                          RNN_TYPE,
                          RNN_LAYER_COUNT,
                          RNN_OUT,
                          RNN_DROPOUT,
                          USE_SPATIAL_DROPOUT,
                          SPATIAL_DROPOUT,
                          LEARNING_RATE)

    print(model.summary())

    start_time = time.time()
    history = model.fit(X_train, Y_train, validation_split=0.333, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=2)
    fit_time = time.time() - start_time

    loss, mae, acc = evaluate_model(model, history, X_test, Y_test, BATCH_SIZE)

    results_dict = {}
    results_dict['EMBED_OUTPUT_DIM'] = EMBED_OUTPUT_DIM
    results_dict['USE_SPATIAL_DROPOUT'] = USE_SPATIAL_DROPOUT
    results_dict['SPATIAL_DROPOUT'] = SPATIAL_DROPOUT

    # RNN_TYPE_DICT = {0:'gru',1:'lstm'}
    if (int(round(RNN_TYPE)) == 0):
        results_dict['RNN_TYPE'] = 'GRU'
    else:
        results_dict['RNN_TYPE'] = 'LSTM'

    results_dict['RNN_LAYER_COUNT'] = RNN_LAYER_COUNT
    results_dict['RNN_OUT'] = RNN_OUT
    results_dict['RNN_DROPOUT'] = RNN_DROPOUT
    results_dict['RECURRENT_DROPOUT'] = RECURRENT_DROPOUT
    results_dict['BATCH_SIZE'] = BATCH_SIZE
    results_dict['LEARNING_RATE'] = LEARNING_RATE
    results_dict['EPOCH'] = EPOCH
    results_dict['DATA_SIZE'] = DATA_SIZE
    results_dict['METRIC'] = METRIC

    results_dict['loss'] = loss
    results_dict['mae'] = mae
    results_dict['acc'] = acc
    results_dict['fit_time'] = fit_time

    model_uuid = uuid.uuid4().hex
    results_dict['model_uuid'] = model_uuid

    results_dict['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

    model.save('./models/' + model_uuid + '.h5')
    with open('./models/' + model_uuid + '.history.json', 'w') as f:
        json.dump(history.history, f)

    # metric for Baysean Optimization
    # thing to minimize:
    # Mean of
    #   loss
    #   1.00-accuracy
    #   secs_to_run / 600 seconds
    bayes_metric = np.mean(np.array([1 - loss, 1 - mae, acc * 100, 600 - fit_time]))
    results_dict['bayes_metric'] = bayes_metric

    results_df = pd.DataFrame.from_dict([results_dict], orient='columns')
    with open('results.csv', 'a') as f:
        results_df.to_csv(f, header=False)

    return bayes_metric


def baysean_param_search():
    from bayes_opt import BayesianOptimization

    pbounds = {
        'DATA_SIZE': (10000.0, 10000.1),
        'METRIC': (1, 3),
        'EMBED_OUTPUT_DIM': (8, 512),
        'USE_SPATIAL_DROPOUT': (0, 1),
        'SPATIAL_DROPOUT': (0.0, 0.1),
        'RNN_TYPE': (0, 1),  # GRU / LSTM
        'RNN_LAYER_COUNT': (0, 2),
        'RNN_OUT': (4, 256),
        'RNN_DROPOUT': (0.0, 0.33),
        'RECURRENT_DROPOUT': (0.0, 0.33),
        'EPOCH': (3, 6),
        'BATCH_SIZE': (4, 256),
        'LEARNING_RATE': (0.0001, 0.5)
    }

    import numpy as np
    _bounds = np.array([item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])], dtype=np.float)

    optimizer = BayesianOptimization(
        f=run_experiment,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=1000, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)


baysean_param_search()
