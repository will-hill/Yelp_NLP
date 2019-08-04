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


def model_factory(X_train, VOCAB_SIZE, EMBED_OUTPUT_DIM, LSTM_LAYER_COUNT, LSTM_OUT, LSTM_DROPOUT, RECURRENT_DROPOUT, USE_SPATIAL_DROPOUT, SPATIAL_DROPOUT):
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

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

    if LSTM_LAYER_COUNT > 1:
        for i in range(LSTM_LAYER_COUNT):
            model.add(LSTM(LSTM_OUT, return_sequences=True, dropout=LSTM_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))

    model.add(LSTM(LSTM_OUT, dropout=LSTM_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'accuracy'])
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


def run_experiment(DATA_SIZE, METRIC, EMBED_OUTPUT_DIM, LSTM_LAYER_COUNT, LSTM_OUT, LSTM_DROPOUT, RECURRENT_DROPOUT, USE_SPATIAL_DROPOUT, SPATIAL_DROPOUT,
                   EPOCHS, BATCH_SIZE):
    import time

    df = get_data(DATA_SIZE, METRIC)
    X, Y, VOCAB_SIZE, X_train, X_test, Y_train, Y_test = data_prep(df, METRIC)

    model = model_factory(X_train, VOCAB_SIZE, EMBED_OUTPUT_DIM, LSTM_LAYER_COUNT, LSTM_OUT, LSTM_DROPOUT, RECURRENT_DROPOUT, USE_SPATIAL_DROPOUT,
                          SPATIAL_DROPOUT)
    print(model.summary())

    start_time = time.time()
    history = model.fit(X_train, Y_train, validation_split=0.333, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    fit_time = time.time() - start_time

    loss, mae, acc = evaluate_model(model, history, X_test, Y_test, BATCH_SIZE)

    return loss, mae, acc, fit_time, model


def param_grid():
    import uuid
    import pandas as pd
    from datetime import datetime

    DATA_SIZES = [10, 100]
    METRICS = ['stars', 'funny', 'useful', 'cool']

    # NN
    EMBED_OUTPUT_DIMS = [8, 16]  # 128
    USE_SPATIAL_DROPOUTS = [False, True]
    SPATIAL_DROPOUTS = [0.0, 0.1]
    LSTM_LAYER_COUNTS = [1, 2]
    LSTM_OUTS = [8, 16]  # 196
    LSTM_DROPOUTS = [0.0, 0.1]
    RECURRENT_DROPOUTS = [0.0, 0.1]

    # INDUCTION
    EPOCHS = [1, 2]
    BATCH_SIZES = [64, 128]
    LEARNING_RATES = [0.1, 0.001]

    ####################
    ### BIG ASS LOOP ###
    ####################
    results_dict = {}

    for EMBED_OUTPUT_DIM in EMBED_OUTPUT_DIMS:
        for USE_SPATIAL_DROPOUT in USE_SPATIAL_DROPOUTS:
            for SPATIAL_DROPOUT in SPATIAL_DROPOUTS:
                for LSTM_LAYER_COUNT in LSTM_LAYER_COUNTS:
                    for LSTM_OUT in LSTM_OUTS:
                        for LSTM_DROPOUT in LSTM_DROPOUTS:
                            for RECURRENT_DROPOUT in RECURRENT_DROPOUTS:
                                for BATCH_SIZE in BATCH_SIZES:
                                    for LEARNING_RATE in LEARNING_RATES:
                                        for EPOCH in EPOCHS:
                                            for DATA_SIZE in DATA_SIZES:
                                                for METRIC in METRICS:
                                                    results_dict['EMBED_OUTPUT_DIM'] = EMBED_OUTPUT_DIM
                                                    results_dict['USE_SPATIAL_DROPOUT'] = USE_SPATIAL_DROPOUT
                                                    results_dict['SPATIAL_DROPOUT'] = SPATIAL_DROPOUT
                                                    results_dict['LSTM_LAYER_COUNT'] = LSTM_LAYER_COUNT
                                                    results_dict['LSTM_OUT'] = LSTM_OUT
                                                    results_dict['LSTM_DROPOUT'] = LSTM_DROPOUT
                                                    results_dict['RECURRENT_DROPOUT'] = RECURRENT_DROPOUT
                                                    results_dict['BATCH_SIZE'] = BATCH_SIZE
                                                    results_dict['LEARNING_RATE'] = LEARNING_RATE
                                                    results_dict['EPOCH'] = EPOCH
                                                    results_dict['DATA_SIZE'] = DATA_SIZE
                                                    results_dict['METRIC'] = METRIC

                                                    loss, mae, acc, fit_time, model = run_experiment(DATA_SIZE,
                                                                                                     METRIC,
                                                                                                     EMBED_OUTPUT_DIM,
                                                                                                     LSTM_LAYER_COUNT,
                                                                                                     LSTM_OUT,
                                                                                                     LSTM_DROPOUT,
                                                                                                     RECURRENT_DROPOUT,
                                                                                                     USE_SPATIAL_DROPOUT,
                                                                                                     SPATIAL_DROPOUT,
                                                                                                     EPOCH,
                                                                                                     BATCH_SIZE)

                                                    results_dict['loss'] = loss
                                                    results_dict['mae'] = mae
                                                    results_dict['acc'] = acc
                                                    results_dict['fit_time'] = fit_time

                                                    model_uuid = uuid.uuid4().hex
                                                    results_dict['model_uuid'] = model_uuid

                                                    results_dict['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

                                                    model.save('./models/' + model_uuid + '.h5')

                                                    results_df = pd.DataFrame.from_dict([results_dict], orient='columns')
                                                    with open('results.csv', 'a') as f:
                                                        results_df.to_csv(f, header=False)


param_grid()
