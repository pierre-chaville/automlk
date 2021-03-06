import logging

log = logging.getLogger(__name__)

try:
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU, LeakyReLU
    from keras.optimizers import Adagrad, Adadelta, RMSprop, Adam
    from keras.layers.core import Dense
    from keras.utils import to_categorical

    import_keras = True
except:
    import_keras = False
    log.info('could not import keras. Neural networks will not be used')


def keras_create_model(params, problem_type):
    # creates a neural net model with params definition

    log.info('creating NN structure')
    model = Sequential()
    for l in range(int(params['number_layers'])):
        if l == 0:
            model.add(Dense(units=params['units'], input_dim=params['input_dim']))
        else:
            model.add(Dense(units=params['units']))
        model.add(Activation(params['activation']))
        if params['batch_normalization']:
            model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))

    model.add(Dense(params['output_dim']))

    if problem_type == 'classification':
        model.add(Activation('sigmoid'))

    keras_compile_model(model, params, problem_type)
    return model


def keras_compile_model(model, params, problem_type):
    # compile the model (usefull to reset weights also)
    log.info('compiling NN model')
    if params['optimizer'] == 'Adagrad':
        optimizer = Adagrad(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adadelta':
        optimizer = Adadelta(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adam':
        optimizer = Adam(lr=params['learning_rate'])
    else:
        optimizer = RMSprop(lr=params['learning_rate'])

    if problem_type == 'regression':
        loss = 'mse'
    elif params['output_dim'] == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model.compile(loss=loss, optimizer=optimizer)
