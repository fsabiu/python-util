import sklearn
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, recall_score
import tensorflow as tf
from tensorflow.keras.layers import Dense

def make_report(model, params, metrics):
    
    # Starting MLFlow run
    mlflow.start_run(run_name=str(params))

    # Logging model parameters
    mlflow.log_params(params)

    # Logging model metrics
    mlflow.log_metrics(metrics)

    # Ending MLFlow run
    mlflow.end_run()

    return

def model_creation(hidden_layers, hidden_units, act_function, learning_rate, optimizer, size=None):
    model = tf.keras.models.Sequential()
    if size is not None:
        model.add(tf.keras.Input(shape=(size,)))

    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation = act_function))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    return model

def model_evaluation(modelType, model, X_test, y_test):

    metrics = {}
    y_pred = None

    if (modelType == 'NN'):
        y_pred = model.predict_classes(X_test)

    if (modelType == 'RF'):
        y_pred = model.predict(X_test)

    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Adding precision, recall, f1 and support for each class + overall <accuracy, macro avg, weighted avg>
    for class_name in report.keys():
        if(isinstance(report[class_name], dict)):
            for metric in report[class_name].keys():
                metrics['-'.join([str(class_name), metric])] = report[class_name][metric]
        else:
            metrics[class_name] = report[class_name]
    
    return metrics

def model_training(model, X_train, y_train, X_val, y_val, pool_size, batch_size, epochs, callbacks, logdir):

    model.fit(X_train, y_train, 
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (X_val, y_val),
        callbacks = callbacks
    )

    """if not (logdir == False):
        model.save_weights(logdir+"model", save_format="h5")
    """
    model.summary()

    return model