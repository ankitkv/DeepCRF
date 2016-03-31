import collections
from os.path import join as pjoin

# file locations
git_dir = ''

data_dir = pjoin(git_dir, 'Data/crf_data/')

log_dir = '/tmp/deepcrf_logs'

train_file = pjoin(data_dir, 'semeval_train/crfpp_text_batch_1.txt')
dev_file = pjoin(data_dir, 'semeval_dev/crfpp_text_batch_1.txt')
vecs_file = pjoin(git_dir, 'Data/crf_data/semeval_vecs.dat')

vis_file = pjoin(git_dir, 'visual.pickle')
embeddings_file = pjoin(git_dir, 'embeddings.model')
directs_file = pjoin(git_dir, 'directs.model')

# feature names and tag list
features = ['word', 'lemma', 'pos', 'normal', 'word_length',
            'prefix', 'suffix', 'all_caps', 'capitalized', 'word_pos',
            'sentence_pos', 'sentence_length', 'med_prefix',
            'umls_match_tag_full', 'umls_match_tag_prefix',
            'umls_match_tag_acro', 'label']

tag_list = ['<P>', 'B', 'Bp', 'I', 'Ip', 'In', 'ID', 'O', 'OD']

input_features = collections.OrderedDict({'pos': 10,
                 'normal': 20, 'prefix': 15, 'suffix': 15,
                 'all_caps': 1, 'capitalized': 1, 'med_prefix': 15})

direct_features = collections.OrderedDict({'umls_match_tag_full': 'O',
                  'umls_match_tag_prefix': 'O', 'umls_match_tag_acro': 'O'})

config = Config(input_features=input_features, direct_features=direct_features,
                tag_list=tag_list)

config.crf_obj_weight = 1.0
config.nn_obj_weight = -1

config.conv_dim = [50,200]
config.conv_window = [5,5]
config.conv_dropout = [True, True]

config.direct_window_size = 3

config.l1_list = [f for f in
        ('word', 'lemma', 'normal', 'prefix', 'suffix', 'med_prefix')
    if f in input_features]
config.l1_reg = 1e-2
config.l1_direct_reg = 5e-3

config.dropout_keep_prob = 0.75

config.learning_rate = 0.001

config.gradient_clip = 50 #5
config.param_clip = 250 #50

config.num_epochs = 10
config.patience_increase = 2

config.optimizer = 'adam'
config.batch_size = 30
