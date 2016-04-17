from pprint import pprint
import os
import argparse
import sys
import shutil
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

    train_data = train_data[:50] # TODO remove
    dev_data = dev_data[:20]     # TODO remove
    test_data = test_data[:20]   # TODO remove

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
    trainf1_holder = tf.Variable(0.0, name='trainf1')
    devf1_holder = tf.Variable(0.0, name='devf1')
    testf1_holder = tf.Variable(0.0, name='testf1')
    l1norm = tf.Variable(0.0, name='l1norm')
    tf.scalar_summary('trainf1', trainf1_holder)
    tf.scalar_summary('devf1', devf1_holder)
    tf.scalar_summary('testf1', testf1_holder)
    tf.scalar_summary('l1norm', l1norm)
    try:
        shutil.rmtree(log_dir)
    except OSError:
        pass
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(log_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    embeddings_saver = tf.train.Saver(params_crf.embeddings)
    if params_crf.direct_un:
        if params_crf.direct_bin:
            params_crf.direct_un.update(params_crf.direct_bin)
        directs_saver = tf.train.Saver(params_crf.direct_un)

    # (accuracies, preds) = train_model(train_data, dev_data, crf, config,
    #                                                       params_crf)

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
        (_,l1) = crf.train_epoch(train_data_ready, config, params_crf)
        visualization['featmap'] = config.feature_maps
        print 'validating', i, '\t', str(datetime.now())
        train_acc = crf.validate_accuracy(train_data_ready, config)
        dev_acc = crf.validate_accuracy(dev_data_ready, config)
        print 'train_acc', train_acc, 'dev_acc', dev_acc
        print 'tagging', i, '\t', str(datetime.now())
        preds = tag_dataset(train_data, config, params_crf, crf)
        sentences = preds_to_sentences(preds, config)
        print 'train epoch', i, '\t', str(datetime.now())
        train_f1, visual = evaluate(sentences, 0.5)
        visualization['train'] = visual
        preds = tag_dataset(dev_data, config, params_crf, crf)
        sentences = preds_to_sentences(preds, config)
        print 'dev epoch', i, '\t', str(datetime.now())
        f1, visual = evaluate(sentences, 0.5)
        visualization['dev'] = visual
        preds = tag_dataset(test_data, config, params_crf, crf)
        sentences = preds_to_sentences(preds, config)
        print 'test epoch', i, '\t', str(datetime.now())
        test_f1, visual = evaluate(sentences, 0.5)
        visualization['test'] = visual
        summary = sess.run(merged, feed_dict={
            trainf1_holder: train_f1,
            devf1_holder: f1,
            testf1_holder: test_f1,
            l1norm: l1})
        writer.add_summary(summary, i)

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
            embeddings_saver.save(sess, embeddings_file)
            print 'Wrote embeddings to', embeddings_file
            if params_crf.direct_un:
                directs_saver.save(sess, directs_file)
                print 'Wrote directs to', directs_file
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
    sess.close()


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
