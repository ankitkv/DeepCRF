import collections
from os.path import join as pjoin

# file locations
git_dir = '/home/jernite/Code/DeepCRF'

data_dir = pjoin(git_dir, 'Data/crf_data_overlaps/')

train_file = pjoin(data_dir, 'semeval_train/crfpp_text_batch_1.txt')
dev_file = pjoin(data_dir, 'semeval_dev/crfpp_text_batch_1.txt')
vecs_file = pjoin(git_dir, 'Data/crf_data_overlaps/semeval_vecs.dat')

vis_file = pjoin(git_dir, 'visual.pickle')
model_file = pjoin(git_dir, 'recent.model')

# feature names and tag list
features = ['word', 'lemma', 'pos', 'normal', 'word_length',
            'prefix', 'suffix', 'all_caps', 'capitalized', 'word_pos',
            'sentence_pos', 'sentence_length', 'med_prefix',
            'umls_match_tag_full', 'umls_match_tag_prefix',
            'umls_match_tag_acro', 'label']

tag_list = ['<P>', 'B', 'Bp', 'I', 'Ip', 'In', 'ID', 'O', 'OD']

input_features = collections.OrderedDict({'word': 200, 'lemma': 200, 'pos': 4,
                 'normal': 200, 'word_length': 1, 'prefix': 100, 'suffix': 100,
                 'capitalized': 1, 'word_pos': 4, 'sentence_pos': 4,
                 'sentence_length': 1, 'med_prefix': 100,
                 'umls_match_tag_full': 3, 'umls_match_tag_prefix': 3,
                 'umls_match_tag_acro': 3})

config = Config(input_features=input_features, tag_list=tag_list)

config.dropout_keep_prob = 0.5

config.learning_rate = 5e-4

config.gradient_clip = 5
config.param_clip = 50

config.num_epochs = 9
config.patience_increase = 1.8

config.optimizer = 'adam'
config.batch_size = 50
