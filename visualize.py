import argparse
import cPickle as pickle
import sys

vis_file = 'visual.pickle'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def read_visualization():
    vis = None
    try:
        with open(vis_file, 'rb') as f:
            vis = pickle.load(f)
    except IOError:
        print >> sys.stderr, 'Could not read visualization file', vis_file
    except pickle.UnpicklingError:
        print >> sys.stderr, 'Could not unpickle visualization!'
    return vis


def visualize(visualization, section, what):
    if what == 'none':
        return
    print
    print
    print bcolors.OKBLUE + 'visualizing', section + bcolors.ENDC
    print
    visual = visualization[section]
    for (fp, fn, sent) in visual:
        if what == 'all' or (what == 'wrong' and (fp > 0 or fn > 0)):
            for word in sent:
                if word['true_mention']:
                    if word['pred']:
                        print bcolors.OKGREEN + word['word'] + bcolors.ENDC,
                    else:
                        print bcolors.WARNING + word['word'] + bcolors.ENDC,
                else:
                    if word['pred']:
                        print bcolors.FAIL + word['word'] + bcolors.ENDC,
                    else:
                        print word['word'],
            print


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the models for \
                                     various parameter values')
    parser.add_argument("-file", "--visualization_file",
                        help="visualization file [default "+vis_file+"]")
    parser.add_argument("-train", "--train",
                        help="one of 'all', 'wrong', 'none'")
    parser.add_argument("-test", "--test",
                        help="one of 'all', 'wrong', 'none'")
    parser.add_argument("-dev", "--dev",
                        help="one of 'all', 'wrong', 'none'")
    args = parser.parse_args()
    if args.visualization_file:
        vis_file = args.visualization_file
    visualization = read_visualization()
    if visualization:
        visualize(visualization, 'train', args.train or 'none')
        visualize(visualization, 'dev', args.dev or 'none')
        visualize(visualization, 'test', args.test or 'none')
