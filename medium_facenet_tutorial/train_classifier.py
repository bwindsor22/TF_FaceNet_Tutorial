import argparse
import logging
import os
import pickle
import sys
import time
import datetime

import pdb

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from lfw_input import filter_dataset, split_dataset, get_dataset
from medium_facenet_tutorial import lfw_input

logger = logging.getLogger(__name__)


def main(input_directory, model_path, output_path, batch_size, num_threads, num_epochs,
         min_images_per_labels, split_ratio, is_append=True, is_train=True):
    
    """
    Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
     a classifier outputted to :param output_path
     
    :param input_directory: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param classifier_output_path: Path to write pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_labels: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    :param is_train: bool denoting if training or evaluate
    """

    start_time = time.time()
    """
    Most TensorFlow programs start with a dataflow graph construction phase. 
    In this phase, you invoke TensorFlow API functions that construct new tf.Operation (node) and tf.Tensor (edge) 
    objects and add them to a tf.Graph instance.
    
    Completed in 6790.907581329346 seconds
    INFO:root:Saved classifier model to file "/medium-facenet-tutorial/output/classifier.pkl"
    """
    
    emb_array, label_array, class_names = _run_tensorflow(input_directory, model_path, output_path, batch_size, num_threads, num_epochs,
                                                            min_images_per_labels, split_ratio, is_train)
    
    if(is_append):
        emb_array, label_array, class_names = _combine_with_previous_embeddings(emb_array, label_array, class_names, output_path)
   
    if( is_train ):
        _record_new_embs(emb_array, label_array, class_names, output_path)
    
    _eval_or_train_svc(emb_array, label_array, class_names, output_path, is_train)
    

    _log_finish(time.time() - start_time)
    

def _log_finish( seconds ):
    logger.info('Completed in HH:MM:SS - {}'.format(str(datetime.timedelta(seconds=seconds))))


def _combine_with_previous_embeddings(emb_array, label_array, class_names, output_path):
    logger.info("adding new entry to csvs")
    
    file = output_path + 'emb_array.csv'
    old_emb_array   = pd.read_csv( file ).values;
    

    emb_array = np.concatenate((old_emb_array, emb_array), axis=0)
    
    file = output_path + 'label_array.csv'
    old_label_array = pd.read_csv( file );
    label_max = old_label_array.max()
    offset = label_max.values[0] + 1

    label_array = label_array + offset
    old_label_array = old_label_array.values

    label_array = np.concatenate((old_label_array, label_array.reshape(-1, 1)), axis=0)
    
    file = output_path + 'class_names.csv'
    old_class_names = pd.read_csv( file ).values;

    class_names = np.concatenate((old_class_names, np.array(class_names).reshape(-1, 1)), axis=0).tolist()
    
    return emb_array, label_array, class_names

def _record_new_embs(emb_array, label_array, class_names, output_path):
    _to_csv(emb_array, 'emb_array', output_path)
    _to_csv(label_array, 'label_array', output_path)
    _to_csv(class_names, 'class_names', output_path)
    
def _to_csv(np_array, file_name,  out_dir):
    logger.info("saving {}".format(file_name))
    df = pd.DataFrame(np_array)
    df.to_csv(out_dir + file_name + ".csv", index=False)


def _run_tensorflow(input_directory, model_path, classifier_output_path, batch_size, num_threads, num_epochs,
                      min_images_per_labels, split_ratio, is_train):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_set, test_set = _get_test_and_train_set(input_directory, min_num_images_per_label=min_images_per_labels,
                                                      split_ratio=split_ratio)
        if is_train:
            images, labels, class_names = _load_images_and_labels(train_set, image_size=160, batch_size=batch_size,
                                                                  num_threads=num_threads, num_epochs=num_epochs,
                                                                  random_flip=True, random_brightness=True,
                                                                  random_contrast=True)
        else:
            images, labels, class_names = _load_images_and_labels(test_set, image_size=160, batch_size=batch_size,
                                                                  num_threads=num_threads, num_epochs=1)

        logger.info('images: {}, {} ...'.format(images[0], images[1]))

        _load_model(model_filepath=model_path)
        
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess) 

        # Super long step
        emb_array, label_array = _create_embeddings(embedding_layer, images, labels, images_placeholder,
                                                    phase_train_placeholder, sess)

        coord.request_stop()
        coord.join(threads=threads)
        logger.info('Created {} embeddings'.format(len(emb_array)))

        return emb_array, label_array, class_names



def _eval_or_train_svc(emb_array, label_array, class_names, output_path, is_train):        

        classifier_filename = output_path + 'classifier.pkl'

        if is_train:
            _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename)
        else:
            _evaluate_classifier(emb_array, label_array, classifier_filename)



def _get_test_and_train_set(input_dir, min_num_images_per_label, split_ratio=0.7):
    """
    Load train and test dataset. Classes with < :param min_num_images_per_label will be filtered out.
    :param input_dir: 
    :param min_num_images_per_label: 
    :param split_ratio: 
    :return: 
    """
    dataset = get_dataset(input_dir)
    dataset = filter_dataset(dataset, min_images_per_label=min_num_images_per_label)
    train_set, test_set = split_dataset(dataset, split_ratio=split_ratio)

    return train_set, test_set


def _load_images_and_labels(dataset, image_size, batch_size, num_threads, num_epochs, random_flip=False,
                            random_brightness=False, random_contrast=False):
    class_names = [cls.name for cls in dataset]
    image_paths, labels = lfw_input.get_image_paths_and_labels(dataset)
    images, labels = lfw_input.read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads,
                                         shuffle=False, random_flip=random_flip, random_brightness=random_brightness,
                                         random_contrast=random_contrast)
    logger.info(images)
    return images, labels, class_names


def _load_model(model_filepath):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        logger.error('Missing model file. Exiting')
        logger.info('Model File path: ' + model_filepath )
        sys.exit(-1)


def _create_embeddings(embedding_layer, images, labels, images_placeholder, phase_train_placeholder, sess):
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer: 
    :param images: 
    :param labels: 
    :param images_placeholder: 
    :param phase_train_placeholder: 
    :param sess: 
    :return: (tuple): image embeddings and labels
    """
    emb_array = None
    label_array = None
    try:
        i = 0
        while True:
            batch_images, batch_labels = sess.run([images, labels])
            logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
            emb = sess.run(embedding_layer,
                           feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass

    return emb_array, label_array


def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier')
    
    start_time = time.time()
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)
    logger.info('Completed SVC training in {} seconds'.format(time.time() - start_time))

    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
        
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)


def _evaluate_classifier(emb_array, label_array, classifier_filename):
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))
    if not os.path.exists(classifier_filename):
        raise ValueError('Pickled classifier not found, have you trained first?')

    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)

        predictions = model.predict_proba(emb_array, )
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, label_array))
        print('Accuracy: %.3f' % accuracy)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        help='Input path of data to train on', default=128)
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                        help='Number of threads to utilize for queue')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                        help='Path to output trained classifier model')
    parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', default=0.7,
                        help='Ratio to split train/test dataset')
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=10,
                        dest='min_images_per_class', help='Minimum number of images per class')
    parser.add_argument('--output-path', type=str, action='store', dest='output_path',
                        help='Path to output trained classifier model')
    parser.add_argument('--is-append', action='store_true', dest='is_append', default=False,
                        help='Flag to indicate adding a new user')
    parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')

    args = parser.parse_args()

    main(input_directory=args.input_dir,
         model_path=args.model_path,
         output_path=args.output_path,
         batch_size=args.batch_size,
         num_threads=args.num_threads,
         num_epochs=args.num_epochs,
         min_images_per_labels=args.min_images_per_class,
         split_ratio=args.split_ratio,
         is_append=args.is_append,
         is_train=args.is_train)
