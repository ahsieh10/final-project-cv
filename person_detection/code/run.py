
import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel, VGGModel
from preprocess import Datasets
from utils import ImageLabelingLogger, CustomModelSaver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')

    return parser.parse_args()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.task)

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    else:
        model = VGGModel()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Make checkpoint directory if needed
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()
