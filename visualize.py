from __future__ import division

import argparse
import cPickle as pickle
import sys
import collections
import numpy as np
import scipy.misc
import scipy.spatial.distance
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *

config_file = 'Configs/my_config.py'
vis_file = ''
visualization = None


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def read_visualization():
    global visualization
    try:
        with open(vis_file, 'rb') as f:
            visualization = pickle.load(f)
        return True
    except IOError:
        print >> sys.stderr, 'Could not read visualization file', vis_file
    except pickle.UnpicklingError:
        print >> sys.stderr, 'Could not unpickle visualization!'
    return False


def _stats(n,d):
    if d==0:
        return (n,d,float('nan'))
    else:
        return (n,d,100.0*float(n)/float(d))


def is_discontinuous(mention):
    disc = False
    for i,tok in enumerate(mention,1):
        if mention[i-1]+1 != tok:
            disc = True
            break
    return disc


def visualize_stats(section, longc_n=2, morec_n=2):
    global visualization
    visual = visualization[section]
    noc_sentences = 0
    noc_mistakes = 0
    nopred_sentences = 0
    nopred_mistakes = 0
    singlec_sentences = 0
    singlec_mistakes = 0
    morec_sentences = 0
    morec_mistakes = 0
    morec_mentions = 0
    morec_mention_mistakes = 0
    morec_preds = 0
    morec_pred_mistakes = 0
    longc_sentences = 0
    longc_mentions = 0
    longc_preds = 0
    longc_sent_mistakes = 0
    longc_mention_mistakes = 0
    longc_pred_mistakes = 0
    discont_sentences = 0
    discont_mentions = 0
    discont_preds = 0
    discont_sent_mistakes = 0
    discont_mention_mistakes = 0
    discont_pred_mistakes = 0
    overlap_sentences = 0
    overlap_mentions = 0
    overlap_preds = 0
    overlap_sent_mistakes = 0
    overlap_mention_mistakes = 0
    overlap_pred_mistakes = 0
    for (sentence, _, _, _, true_mentions, preds, fp, fn) in visual:
        mistaken = fp > 0 or fn > 0
        morec = False
        if not true_mentions:
            noc_sentences += 1
            if mistaken:
                noc_mistakes += 1
        if not preds:
            nopred_sentences += 1
            if mistaken:
                nopred_mistakes += 1
        if len(true_mentions) == 1:
            singlec_sentences += 1
            if mistaken:
                singlec_mistakes += 1
        if len(true_mentions) > morec_n:
            morec = True
            morec_sentences += 1
            if mistaken:
                morec_mistakes += 1
        longc = False
        discont = False
        overlap = False
        for mention in true_mentions:
            mention_mistaken = mention not in preds
            if morec:
                morec_mentions += 1
                if mention_mistaken:
                    morec_mention_mistakes += 1
            if len(mention) > longc_n:
                longc = True
                longc_mentions += 1
                if mention_mistaken:
                    longc_mention_mistakes += 1
            if is_discontinuous(mention):
                discont = True
                discont_mentions += 1
                if mention_mistaken:
                    discont_mention_mistakes += 1
            mention_overlap = False
            for altmention in true_mentions:
                if altmention != mention:
                    if altmention[-1] > mention[0] and \
                                mention[-1] > altmention[0]:
                        mention_overlap = True
                        break
            if mention_overlap:
                overlap = True
                overlap_mentions += 1
                if mention_mistaken:
                    overlap_mention_mistakes += 1
        for pred in preds:
            pred_mistaken = pred not in true_mentions
            if morec:
                morec_preds += 1
                if pred_mistaken:
                    morec_pred_mistakes += 1
            if len(pred) > longc_n:
                longc_preds += 1
                if pred_mistaken:
                    longc_pred_mistakes += 1
            if is_discontinuous(pred):
                discont_preds += 1
                if pred_mistaken:
                    discont_pred_mistakes += 1
            pred_overlap = False
            for altpred in preds:
                if altpred != pred:
                    if altpred[-1] > pred[0] and pred[-1] > altpred[0]:
                        pred_overlap = True
                        break
            if pred_overlap:
                overlap_preds += 1
                if pred_mistaken:
                    overlap_pred_mistakes += 1
        if longc:
            longc_sentences += 1
            if mistaken:
                longc_sent_mistakes += 1
        if discont:
            discont_sentences += 1
            if mistaken:
                discont_sent_mistakes += 1
        if overlap:
            overlap_sentences += 1
            if mistaken:
                overlap_sent_mistakes += 1
    print
    print 'Incorrect sentences (no concepts):               %d/%d   (%f)' \
                % _stats(noc_mistakes, noc_sentences)
    print 'Incorrect sentences (no predictions):            %d/%d   (%f)' \
                % _stats(nopred_mistakes, nopred_sentences)
    print 'Incorrect sentences (single concept):            %d/%d   (%f)' \
                % _stats(singlec_mistakes, singlec_sentences)
    print 'Incorrect sentences (more than %d concepts):      %d/%d   (%f)' \
                % ((morec_n,) + _stats(morec_mistakes, morec_sentences))
    print 'Incorrect sentences (concepts longer than %d):    %d/%d   (%f)' \
                % ((longc_n,) + _stats(longc_sent_mistakes, longc_sentences))
    print 'Incorrect sentences (discontinuous concepts):    %d/%d   (%f)' \
                % _stats(discont_sent_mistakes, discont_sentences)
    print 'Incorrect sentences (overlapping concepts):      %d/%d   (%f)' \
                % _stats(overlap_sent_mistakes, overlap_sentences)
    print
    print 'Missed mentions (more than %d concepts):          %d/%d   (%f)' \
                % ((morec_n,) + _stats(morec_mention_mistakes, morec_mentions))
    print 'Missed mentions (concepts longer than %d):        %d/%d   (%f)' \
                % ((longc_n,) + _stats(longc_mention_mistakes, longc_mentions))
    print 'Missed mentions (discontinuous concepts):        %d/%d   (%f)' \
                % _stats(discont_mention_mistakes, discont_mentions)
    print 'Missed mentions (overlapping concepts):          %d/%d   (%f)' \
                % _stats(overlap_mention_mistakes, overlap_mentions)
    print
    print 'Incorrect predictions (more than %d concepts):    %d/%d   (%f)' \
                % ((morec_n,) + _stats(morec_pred_mistakes, morec_preds))
    print 'Incorrect predictions (preds longer than %d):     %d/%d   (%f)' \
                % ((longc_n,) + _stats(longc_pred_mistakes, longc_preds))
    print 'Incorrect predictions (discontinuous preds):     %d/%d   (%f)' \
                % _stats(discont_pred_mistakes, discont_preds)
    print 'Incorrect predictions (overlapping preds):       %d/%d   (%f)' \
                % _stats(overlap_pred_mistakes, overlap_preds)
    print


def visualize_preds(section, what):
    if what == 'stats':
        visualize_stats(section)
        return
    global visualization
    print
    print
    print bcolors.HEADER + 'visualizing', section + bcolors.ENDC
    print
    visual = visualization[section]
    for (sentence, _, _, _, true_mentions, preds, fp, fn) in visual:
        if what == 'all' or (what == 'wrong' and (fp > 0 or fn > 0)):
            print bcolors.OKBLUE + '\nTrue:' + bcolors.ENDC,
            for mention in true_mentions:
                if mention not in preds:
                    print bcolors.WARNING + str(mention) + bcolors.ENDC,
                else:
                    print bcolors.OKGREEN + str(mention) + bcolors.ENDC,
            print bcolors.OKBLUE + '\nPred:' + bcolors.ENDC,
            for pred in preds:
                if pred not in true_mentions:
                    print bcolors.FAIL + str(pred) + bcolors.ENDC,
                else:
                    print bcolors.OKGREEN + str(pred) + bcolors.ENDC,
            preds = set(p for pred in preds for p in pred)
            true_mentions = set(m for mention in true_mentions
                                  for m in mention)
            print bcolors.OKBLUE + '\nIndx:' + bcolors.ENDC,
            for i, word in enumerate(sentence):
                if i in true_mentions:
                    if i in preds:
                        print bcolors.OKGREEN + \
                            str(i).center(len(word['word'])) + bcolors.ENDC,
                    else:
                        print bcolors.WARNING + \
                            str(i).center(len(word['word'])) + bcolors.ENDC,
                else:
                    if i in preds:
                        print bcolors.FAIL + \
                            str(i).center(len(word['word'])) + bcolors.ENDC,
                    else:
                        print str(i).center(len(word['word'])),
            print bcolors.OKBLUE + '\nSent:' + bcolors.ENDC,
            for i, word in enumerate(sentence):
                if i in true_mentions:
                    if i in preds:
                        print bcolors.OKGREEN + \
                            word['word'].center(len(str(i))) + bcolors.ENDC,
                    else:
                        print bcolors.WARNING + \
                            word['word'].center(len(str(i))) + bcolors.ENDC,
                else:
                    if i in preds:
                        print bcolors.FAIL + \
                            word['word'].center(len(str(i))) + bcolors.ENDC,
                    else:
                        print word['word'].center(len(str(i))),
            print


def visualize_activations(args, n=25, rare=0.0001, use_pots=True):
    global visualization
    feature, selected_tag = args
    if feature not in input_features:
        print >> sys.stderr, 'no such feature:', feature + '.', \
                             'choose one of:'
        print >> sys.stderr, ' '.join(input_features.keys())
        return
    if selected_tag == 'all': selected_tag = None
    if selected_tag and selected_tag not in tag_list:
        print >> sys.stderr, 'no such tag:', selected_tag + '.', \
                             'choose one of:'
        print >> sys.stderr, ' '.join(tag_list)
        return
    visual = []
    for section in ('train', 'dev', 'test'):
        visual.extend(visualization[section])
    window = (sum(config.conv_window) - len(config.conv_window) + 1) // 2
    all_pots = collections.defaultdict(lambda: collections.defaultdict(list))
    fd = collections.defaultdict(int)
    for v in visual:
        sentence = v[0]
        for word in sentence:
            fd[word[feature]] += 1
    valid = set(k for k,v in fd.items() if v >= rare * sum(fd.values()))
    for (sentence, un_pots, bin_pots, preds, _, _, _, _) in visual:
        pre_len = (len(un_pots) - len(sentence)) // 2
        for i, (upots, bpots, pred) in enumerate(zip(
                                  un_pots[pre_len:],bin_pots[pre_len:],preds)):
            for j in range(-window, window+1):
                pos = i + j
                if pos >= 0 and pos < len(sentence):
                    value = sentence[pos][feature]
                    if value not in valid:
                        continue
                    if not use_pots:
                        upots = [0.0] * len(tag_list)
                        upots[pred[1]] = 1.0
                        use_bpots = False
                    else:
                        use_bpots = pos > 0
                    all_pots[value][j].append((use_bpots, upots, bpots))
    final = collections.defaultdict(lambda: collections.defaultdict(list))
    for value, positions in all_pots.items():
        for pos, pot_list in positions.items():
            for i in range(len(tag_list)):
                if selected_tag and tag_list[i] != selected_tag: continue
                P = np.array([p for p,u,b in pot_list])
                U = np.array([u for p,u,b in pot_list])
                B = np.array([b for p,u,b in pot_list])
                pot = np.sum(U[:,i]) + np.dot(P,
                                              scipy.misc.logsumexp(B[:,:,i],1))
                pot /= float(len(pot_list))
                final[i][pos].append((pot, value))
    for tag, positions in final.items():
        print
        print bcolors.HEADER+'TAG', tag_list[tag] + ':'+bcolors.ENDC
        print
        for pos, values in sorted(positions.items(), key=lambda x:x[0]):
            if pos == 0:
                color = bcolors.OKGREEN
            else:
                color = bcolors.WARNING
            print
            print bcolors.OKBLUE+'Position', str(pos) + ':'+bcolors.ENDC
            values.sort(key=lambda x:x[0], reverse=True)
            for pot, value in values[:n]:
                print color+str(value)+bcolors.ENDC, '('+str(pot)+')',
            if len(values) > 2*n:
                print
                print ' ... '
            for pot, value in values[n:][-n:]:
                print color+str(value)+bcolors.ENDC, '('+str(pot)+')',
            print
        print


def visualize_embeddings(feature, n=50):
    global visualization
    (feature_name, feature_value) = feature
    feature_mappings = visualization['featmap']
    if feature_name not in input_features:
        print >> sys.stderr, 'no such feature:', feature_name + '.', \
                             'choose one of:'
        print >> sys.stderr, ' '.join(input_features.keys())
        return
    all_values = set(feature_mappings[feature_name]['reverse'])
    while feature_value not in all_values:
        try:
            if type(feature_value) is not int:
                feature_value = int(feature_value)
                continue
        except ValueError:
            pass
        print >> sys.stderr, 'no such value for', feature_name + ':', \
                             feature_value + '. choose one of:'
        print >> sys.stderr, ' '.join([str(e) for e in all_values])
        return
    with tf.device('/cpu:0'):
        sess = tf.InteractiveSession()
        param_vars = {}
        for i, (feat, dim) in enumerate(config.input_features.items()):
            if not dim: continue
            shape = [len(feature_mappings[feat]['reverse']), dim]
            initial = tf.truncated_normal(shape, stddev=0.1)
            emb_matrix = tf.Variable(initial, name=feat + '_embedding')
            param_vars[feat] = emb_matrix
        embeddings_saver = tf.train.Saver(param_vars)
        embeddings_saver.restore(sess, embeddings_file)
        embeddings = param_vars[feature_name].eval()
        target_index = feature_mappings[feature_name]['lookup'][feature_value]
        target_embedding = embeddings[target_index]
        dists = []
        for emb in embeddings:
            dists.append(scipy.spatial.distance.cosine(emb, target_embedding))
        values = zip(dists, embeddings,
                     feature_mappings[feature_name]['reverse'])
        values.sort(key=lambda x:x[0])
        values = [(d,e,v) for (d,e,v) in values if np.linalg.norm(e) >= 1e-6]
        maxnorm = values[-1][0]
        print
        print str(min(n,len(values)-1))+'/'+str(len(values)-1), \
              'closest neighbors of the', feature_name, feature_value + ':'
        print
        for d, e, v in values[1:n+1]:
            print (str(d / maxnorm) + ':').ljust(10), v
        print
        if len(values)-1 > n:
            full = len(values)-1
            values = values[n+1:][-n:]
            print
            print str(min(n,len(values)))+'/'+str(full), \
                 'farthest neighbors of the', feature_name, feature_value + ':'
            print
            for d, e, v in values[:n]:
                print (str(d / maxnorm) + ':').ljust(10), v
            print
        sess.close()


def visualize_sparsity(feature_name, n=200):
    global visualization
    feature_mappings = visualization['featmap']
    if feature_name not in input_features:
        print >> sys.stderr, 'no such feature:', feature_name + '.', \
                             'choose one of:'
        print >> sys.stderr, ' '.join(input_features.keys())
        return
    with tf.device('/cpu:0'):
        sess = tf.InteractiveSession()
        param_vars = {}
        for i, (feat, dim) in enumerate(config.input_features.items()):
            if not dim: continue
            shape = [len(feature_mappings[feat]['reverse']), dim]
            initial = tf.truncated_normal(shape, stddev=0.1)
            emb_matrix = tf.Variable(initial, name=feat + '_embedding')
            param_vars[feat] = emb_matrix
        embeddings_saver = tf.train.Saver(param_vars)
        embeddings_saver.restore(sess, embeddings_file)
        embeddings = param_vars[feature_name].eval()
        print
        print 'Non-zero embeddings: ' + str(sum(1 for emb in embeddings
                  if np.linalg.norm(emb) >= 1e-6)) + '/' + str(len(embeddings))
        print
        print 'Top embeddings:'
        values = zip(embeddings, feature_mappings[feature_name]['reverse'])
        values.sort(key=lambda x:np.linalg.norm(x[0]), reverse=True)
        for e, v in values[:n]:
            print (str(np.linalg.norm(e)) + ':').ljust(10), v
        print
        plt.plot([np.linalg.norm(e) for e,v in values])
        plt.show()
        sess.close()


def visualize_directs(feature_name, thres=0.025):
    global visualization
    feature_mappings = visualization['featmap']
    if feature_name not in direct_features:
        print >> sys.stderr, 'no direct feature:', feature_name + '.', \
                             'choose one of:'
        print >> sys.stderr, ' '.join(direct_features.keys())
        return
    with tf.device('/cpu:0'):
        sess = tf.InteractiveSession()
        directs_bin = collections.OrderedDict({})
        for (idx,feat) in enumerate(config.direct_features):
            i = idx + len(config.input_features) - len(config.direct_features)
            shape = [len(feature_mappings[feat]['reverse']), config.n_tags ** 2]
            initial = tf.zeros(shape)
            direct_matrix = tf.Variable(initial, name=feat + '_bin_direct')
            directs_bin[feat + '_bin_direct'] = direct_matrix
        directs_un = collections.OrderedDict({})
        for (idx,feat) in enumerate(config.direct_features):
            i = idx + len(config.input_features) - len(config.direct_features)
            shape = [len(feature_mappings[feat]['reverse']), config.n_tags]
            initial = tf.zeros(shape)
            direct_matrix = tf.Variable(initial, name=feat + '_un_direct')
            directs_un[feat + '_un_direct'] = direct_matrix
        directs_un.update(directs_bin)
        directs_saver = tf.train.Saver(directs_un)
        directs_saver.restore(sess, directs_file)
        unary_matrix = directs_un[feature_name + '_un_direct'].eval()
        binary_matrix = directs_bin[feature_name + '_bin_direct'].eval()
        print
        print 'BINARY'
        print
        print ''.ljust(10),
        for tag1 in tag_list:
            for tag2 in tag_list:
                print bcolors.HEADER+(tag1+':'+tag2).ljust(10)+bcolors.ENDC,
        print
        print
        for i,row in enumerate(binary_matrix):
            print bcolors.OKBLUE+feature_mappings[feature_name]['reverse'][i]\
                                                       .ljust(10)+bcolors.ENDC,
            for e in row:
                if e < -thres:
                    print bcolors.FAIL+("% .4f"%e).ljust(10)+bcolors.ENDC,
                elif e > thres:
                    print bcolors.OKGREEN+("% .4f"%e).ljust(10)+bcolors.ENDC,
                else:
                    print ("% .4f"%e).ljust(10),
            print
            print
        print
        print 'UNARY'
        print
        print ''.ljust(10),
        for tag in tag_list:
            print bcolors.HEADER+('  '+tag).ljust(10)+bcolors.ENDC,
        print
        for i,row in enumerate(unary_matrix):
            print bcolors.OKBLUE+feature_mappings[feature_name]['reverse'][i].\
                                                        ljust(10)+bcolors.ENDC,
            for e in row:
                if e < -thres:
                    print bcolors.FAIL+("% .4f"%e).ljust(10)+bcolors.ENDC,
                elif e > thres:
                    print bcolors.OKGREEN+("% .4f"%e).ljust(10)+bcolors.ENDC,
                else:
                    print ("% .4f"%e).ljust(10),
            print
        print
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the models for \
                                     various parameter values')
    parser.add_argument("-conf", "--config_file",
                        help="location of configuration file")
    parser.add_argument("-file", "--visualization_file",
                        help="visualization file")
    parser.add_argument("-train", "--train",
                        help="one of 'all', 'wrong' or 'stats'")
    parser.add_argument("-test", "--test",
                        help="one of 'all', 'wrong' or 'stats'")
    parser.add_argument("-dev", "--dev",
                        help="one of 'all', 'wrong' or 'stats'")
    parser.add_argument("-pots", "--potentials", nargs=2,
                        metavar=("FEATURE_NAME", "TAG"),
                        help="feature values that lead to high unary pots. " \
                             "TAG can be 'all'.")
    parser.add_argument("-outs", "--outputs", nargs=2,
                        metavar=("FEATURE_NAME", "TAG"),
                        help="feature values that lead to outputs. " \
                             "TAG can be 'all'.")
    parser.add_argument("-embed", "--embed", nargs=2,
                        metavar=("FEATURE_NAME", "FEATURE_VALUE"),
                        help="feature name and feature value")
    parser.add_argument("-sparsity", "--sparsity", metavar="FEATURE_NAME",
                        help="sparsity in the embeddings")
    parser.add_argument("-directs", "--directs", metavar="FEATURE_NAME",
                        help="mapping matrix for upper-level feature")
    args = parser.parse_args()
    if args.config_file:
        config_file = os.path.abspath(args.config_file)
    execfile(config_file)
    if read_visualization():
        if args.visualization_file:
            vis_file = args.visualization_file
        if args.train:
            visualize_preds('train', args.train)
        if args.dev:
            visualize_preds('dev', args.dev)
        if args.test:
            visualize_preds('test', args.test)
        if args.potentials:
            visualize_activations(args.potentials, use_pots=True)
        if args.outputs:
            visualize_activations(args.outputs, use_pots=False)
        if args.embed:
            visualize_embeddings(args.embed)
        if args.sparsity:
            visualize_sparsity(args.sparsity)
        if args.directs:
            visualize_directs(args.directs)

