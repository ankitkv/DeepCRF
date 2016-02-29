from pprint import pprint
import os
import argparse
import sys
from datetime import datetime
import cPickle as pickle

from crf_defs import *

config_file = 'Configs/my_config.py'
config = None

train_file = ''
dev_file = ''
features = []


def write_visualization(visual):
    try:
        with open(vis_file, 'wb') as f:
            pickle.dump(visual, f, -1)
            print 'Wrote visualization info to', vis_file
    except IOError:
        print >> sys.stderr, 'Could not write visualization to file', vis_file
    except pickle.PicklingError:
        print >> sys.stderr, 'Could not pickle visualization!'


def main():
    # load the data
    train_data, dev_data = read_data(train_file, features, config, 11000)
    test_data = read_data(dev_file, features, config)
    config.make_mappings(train_data + dev_data + test_data)
    # initialize the parameters
    if config.init_words:
        word_vectors = read_vectors(vecs_file,
                                    config.feature_maps['word']['reverse'])
        pre_trained = {'word': word_vectors}
    else:
        pre_trained = {}
    params_crf = Parameters(init=pre_trained)
    # make the CRF and SequNN models
    sess = tf.InteractiveSession()
    crf = CRF(config)
    crf.make(config, params_crf)
    sess.run(tf.initialize_all_variables())
    embeddings_saver = tf.train.Saver(params_crf.embeddings)
    # (accuracies, preds) = train_model(train_data, dev_data, crf, config,
    #                                                       params_crf, 'CRF')

    best_f1 = 0.0
    best_train_f1 = 0.0
    best_test_f1 = 0.0
    patience = float(config.num_epochs)
    improvement_th = config.improvement_threshold
    patience_increase = config.patience_increase
    i = 0
    while True:
        visualization = {}
        print
        train_data_ready = prepare_data(train_data, config)
        dev_data_ready = prepare_data(dev_data, config)
        print 'training', i, '\t', str(datetime.now())
        crf.train_epoch(train_data_ready, config, params_crf)
        visualization['featmap'] = config.feature_maps
        print 'validating', i, '\t', str(datetime.now())
        train_acc = crf.validate_accuracy(train_data_ready, config)
        dev_acc = crf.validate_accuracy(dev_data_ready, config)
        print 'train_acc', train_acc, 'dev_acc', dev_acc
        print 'tagging', i, '\t', str(datetime.now())
        preds = tag_dataset(train_data, config, params_crf, 'CRF', crf)
        sentences = preds_to_sentences(preds, config)
        print 'train epoch', i, '\t', str(datetime.now())
        train_f1, visual = evaluate(sentences, 0.5)
        visualization['train'] = visual
        preds = tag_dataset(dev_data, config, params_crf, 'CRF', crf)
        sentences = preds_to_sentences(preds, config)
        print 'dev epoch', i, '\t', str(datetime.now())
        f1, visual = evaluate(sentences, 0.5)
        visualization['dev'] = visual
        preds = tag_dataset(test_data, config, params_crf, 'CRF', crf)
        sentences = preds_to_sentences(preds, config)
        print 'test epoch', i, '\t', str(datetime.now())
        test_f1, visual = evaluate(sentences, 0.5)
        visualization['test'] = visual
        if f1 > best_f1:
            print 'found new best!'
            old_p = patience
            if f1 * improvement_th > best_f1:
                patience = max(patience, patience_increase * i)
                if patience != old_p:
                    print 'increasing patience from', old_p, 'to', patience
            else:
                print 'not a significant improvement'
                if patience <= i+2:
                    patience = patience + 2
                    print 'increasing patience from', old_p, 'to', patience
            best_f1 = f1
            best_train_f1 = train_f1
            best_test_f1 = test_f1
            write_visualization(visualization)
            embeddings_saver.save(sess, model_file)
            print 'Wrote embeddings to', model_file
        print 'best dev F1 is:', best_f1
        print ' with train F1:', best_train_f1
        print '   and test F1:', best_test_f1
        if patience <= i+1:
            print 'out of patience'
            break
        else:
            print 'patience:', patience
        i += 1
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing the models for \
                                     various parameter values')
    parser.add_argument("-conf", "--config_file",
                        help="location of configuration file")
    parser.add_argument("-dropout", "--dropout",
                        help="dropout keep probability")
    args = parser.parse_args()
    if args.config_file:
        config_file = os.path.abspath(args.config_file)
    print 'Starting'
    execfile(config_file)
    if args.dropout:
        dropout = float(args.dropout)
        print 'Using the provided keep_prob value of', dropout
        config.dropout_keep_prob = dropout
    main()
