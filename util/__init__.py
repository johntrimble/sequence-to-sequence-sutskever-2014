from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
from line_profiler import LineProfiler

import os
import re
import io
import gzip
import json
import itertools
import numpy as np
import subprocess
import tempfile

##########################
## Profiling
##########################

# take from https://zapier.com/engineering/profiling-python-boss/
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


##########################
## IO
##########################

def smart_open(file, mode='r', encoding='utf-8'):
    if isinstance(file, str) and file.endswith('.gz'):
        return io.TextIOWrapper(gzip.open(file, mode), encoding=encoding)
    else:
        return open(file, mode=mode, encoding=encoding)

def lines_from_files(*files):
    file_handles = []
    try:
        # open the files
        for path in files:
            file_handles.append(smart_open(path, 'r'))

        # generate lines
        for line in itertools.chain.from_iterable(file_handles):
            yield line
    finally:
        # make sure all file handles are closed
        for f in file_handles:
            try:
                f.close()
            except:
                traceback.print_exc()

def list_files_by_regex(regex, directory='.'):
    return [os.path.join(directory, f) for f in os.listdir(directory) if re.fullmatch(regex, f)]

def convert_format_string_to_regex(format_string):
    parts = [re.escape(part) for part in re.split(r'{[^}]*}', format_string)]
    pattern = '(.+)'.join(parts)
    return pattern


##########################
## Reading tokens
##########################

OOV_TOKEN='<unk>'
OOV_IDX=1

SOS_TOKEN='<s>'
SOS_IDX=2

EOS_TOKEN='</s>'
EOS_IDX=3

def pretokenized_to_tokens(s):
    tokens = s.split()
    tokens.append('</s>')
    return tokens

def tokenized_lines_from_pretokenized_files(*files):
    return map(pretokenized_to_tokens, lines_from_files(*files))


##########################
## BLEU scores
##########################

PATH_TO_MULTI_BLEU='./scripts/multi-bleu.perl'

def _run_multi_bleu(reference_path, hypothesis_path):
    with open(hypothesis_path, 'r') as f:
        with subprocess.Popen([PATH_TO_MULTI_BLEU, reference_path], 
                              stdin=f, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.DEVNULL) as proc:
            return proc.stdout.read().decode('utf-8')
        
def multi_bleu(reference_path, hypothesis_path):
    output = _run_multi_bleu(reference_path, hypothesis_path)
    return float(re.match(r'^BLEU\s*=\s*([0-9.]+).*', output).group(1))

def calc_bleu_scores_by_length_thresholds(source_filename,
                                          reference_filename,
                                          predicted_filename, 
                                          lengths):
    with tempfile.TemporaryDirectory() as temp_directory:        
        # get the length of each line in the source file
        source_lengths = []
        with open(source_filename, mode='r', encoding='utf-8') as f:
            for line in f:
                source_lengths.append(len(line.split(' ')))

        lengths = sorted(lengths)
        length_files = [os.path.join(temp_directory, '{:03d}.lines'.format(length)) 
                        for length in lengths]
        predicted_length_files = [length_file + '.predicted'
                                 for length_file in length_files]
        
        length_file_handles = [open(length_file, mode='w', encoding='utf-8') 
                               for length_file in length_files]
        predicted_length_file_handles = [open(predicted_length_file, mode='w', encoding='utf-8') 
                                         for predicted_length_file in predicted_length_files]

        # partition reference and prediction into files for each length
        try:
            with open(reference_filename, mode='r', encoding='utf-8') as reference_f:
                with open(predicted_filename, mode='r', encoding='utf-8') as predicted_f:
                    params = zip(source_lengths, reference_f, predicted_f)
                    for source_length, reference_line, predicted_line in params:
                        for i in range(len(lengths)):
                            if lengths[i] >= source_length:
                                length_file_handles[i].write(reference_line)
                                predicted_length_file_handles[i].write(predicted_line)
                                break
        finally:
            for f in length_file_handles + predicted_length_file_handles:
                try:
                    f.close()
                except:
                    pass

        # generate bleu scores
        bleu_scores = []
        params = zip(length_files, predicted_length_files)
        for (reference_file, predicted_file) in params:
            if os.stat(reference_file).st_size == 0 and os.stat(predicted_file).st_size == 0:
                bleu_scores.append(None)
                continue
            bleu_score = multi_bleu(reference_file, predicted_file)
            bleu_scores.append(bleu_score)

        return tuple(filter(lambda x: x[1], zip(lengths, bleu_scores)))


##########################
## Bitext Descriptors
##########################

BITEXT_FILENAME_REGEX=r'(?:.*/)?([^/]+)[.](en|fr)([.][^.]*)?'

def parse_bitext_filename(filename):
    m = re.match(BITEXT_FILENAME_REGEX, filename)
    if m:
        return m.group(1), m.group(2)
    else:
        return None

def load_cslm_bitext_descriptors(directory='./target'):
    def to_doc_lang_path(paths):
        doc_lang_path = {}
        for p in paths:
            doc_name, lang = parse_bitext_filename(p)
            if not doc_name in doc_lang_path:
                doc_lang_path[doc_name] = {}
            doc_lang_path[doc_name][lang] = p
        return doc_lang_path

    train_bitext_files_dir = os.path.join(directory, 'bitexts.selected')
    train_bitext_files = list_files_by_regex(
        BITEXT_FILENAME_REGEX,
        directory=train_bitext_files_dir)

    dev_text_bitext_files_dir = os.path.join(directory, 'dev')
    dev_test_bitext_files = list_files_by_regex(
        BITEXT_FILENAME_REGEX,
        directory=dev_text_bitext_files_dir)

    result = {}
    result['train'] = to_doc_lang_path(train_bitext_files)
    result['dev+test'] = to_doc_lang_path(dev_test_bitext_files)
    result['test'] = dict(filter(lambda x: x[0] == 'ntst14', result['dev+test'].items()))
    result['dev'] = dict(filter(lambda x: x[0] != 'ntst14', result['dev+test'].items()))
    return result

def get_bitext_files_from_descriptors_by_lang(descriptors, lang):
    descriptors = sorted(list(descriptors))
    return list(map(lambda descriptor: descriptor[1][lang], descriptors))

def get_bitext_files_from_context_by_lang(context, lang):
    result = []
    descriptors = []
    for doc_lang_path in context.values():
        descriptors.extend(doc_lang_path.items())
    return get_bitext_files_from_descriptors_by_lang(descriptors, lang)

def tokenized_lines_from_bitexts(bitext_descriptors, lang):
    files = get_bitext_files_from_descriptors_by_lang(bitext_descriptors, lang)
    return tokenized_lines_from_pretokenized_files(*files)


##########################
## Build Word Indices
##########################

def build_word_index_from_tokenized_lines(tokenized_lines_iterable):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(tokenized_lines_iterable)
    sorted_words = [OOV_TOKEN, SOS_TOKEN, EOS_TOKEN]
    sorted_words_suffix = [w for (w,i) in sorted(tokenizer.word_index.items(), key=lambda x: (x[1], x[0]))
                           if not w in sorted_words]
    sorted_words.extend(sorted_words_suffix)
    indices = range(1, len(sorted_words)+1)
    word_index = dict(zip(sorted_words, indices))
    return word_index

def build_word_index_from_bitexts(bitext_descriptors, lang):
    tokenized_lines_iter = tokenized_lines_from_bitexts(bitext_descriptors, lang)
    word_index = build_word_index_from_tokenized_lines(tokenized_lines_iter)
    return word_index

def load_word_index(word_index_file):
    with smart_open(word_index_file, 'r') as f:
        return json.load(f)

def build_save_load_word_index(word_index_file, bitext_descriptors, lang, overwrite=False):
    if not os.path.isfile(word_index_file) or overwrite:
        word_index = build_word_index_from_bitexts(bitext_descriptors, lang)
        with smart_open(word_index_file, 'w') as out:
            json.dump(word_index, out)
        return word_index
    else:
        return load_word_index(word_index_file)

def tokenized_lines_to_sequences(word_index, tokenized_lines_iterable):
    tokenizer = Tokenizer(char_level=False, oov_token=OOV_TOKEN)
    tokenizer.word_index = word_index
    return tokenizer.texts_to_sequences(list(tokenized_lines_iterable))

def load_bitext_sequences(bitext_descriptors, word_index, lang):
    tokenized_lines = tokenized_lines_from_bitexts(bitext_descriptors, lang)
    return tokenized_lines_to_sequences(word_index, tokenized_lines)

def apply_vocab_size_constraint(sequences, vocab_size):
    return [[token_idx if token_idx <= vocab_size else OOV_IDX for token_idx in sequence] for sequence in sequences]


##########################
## Sequence Utilities
##########################

class EpochAsPartialIterationSequenceWrapper(Sequence):
    """
    This class allows an iteration over a given sequence of data to be partitioned into
    multiple epcohs during training. This is useful when an iteration takes a particularly
    long time to complete, yet it is desirable that Callbacks that typically trigger at the
    end of an epoch trigger more frequently.
    """
    def __init__(self, wrapped_sequence, epochs_per_iteration):
        if not hasattr(wrapped_sequence, 'reset'):
            raise Exception("The provided `wrapped_sequence` does not implement a required `reset()` method.")
        self.wrapped_sequence = wrapped_sequence
        self.steps_per_epoch = int(len(wrapped_sequence) / epochs_per_iteration)
        self.shuffle_every_n_epochs = epochs_per_iteration
        self.reset()
        
    def reset(self):
        self.wrapped_sequence.reset()
        self.current_epoch = 0
        
    def switch_to_epoch(self, epoch):
        if epoch < self.current_epoch:
            self.reset()
        for _ in range(epoch - self.current_epoch):
            self.on_epoch_end()
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, index):
        index = (self.current_epoch % self.shuffle_every_n_epochs) * self.steps_per_epoch + index
        return self.wrapped_sequence[index]
    
    def on_epoch_end(self):
        self.current_epoch += 1
        if self.current_epoch % self.shuffle_every_n_epochs == 0:
            self.wrapped_sequence.on_epoch_end()

            
##########################
## Weight utilities
##########################

def get_weight_filename_pattern(directory_path):
    model_checkpoint_file = os.path.join(directory_path, 
                                         'weights.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.h5')
    return model_checkpoint_file

def parse_weights_filename(f):
    m = re.match(r'weights[.]([0-9]+)-([0-9.]+)(-([0-9.]+))?[.]h5', f)
    if m:
        val_loss = float(m.group(2)) if m.group(2) else None
        loss = float(m.group(4)) if m.group(4) else None
        return (int(m.group(1)), val_loss, loss)
    
def list_all_weight_files_and_props(directory_path):
    files = os.listdir(directory_path)
    files_and_props = ((os.path.join(directory_path, f), parse_weights_filename(f)) for f in files)
    files_and_props = [(file, props) for (file, props) in files_and_props if props]
    return files_and_props

def latest_epoch_and_weights_file(directory_path):
    files_and_props = list_all_weight_files_and_props(directory_path)
    
    def get_epoch(files_and_props):
        file, (epoch, val_loss, loss) = files_and_props
        return epoch
    
    if not files_and_props:
        return None
    else:
        (file, (epoch, *_)) = sorted(files_and_props, reverse=True, key=get_epoch)[0]
        return (file, epoch)


##########################
## CSLM data
## http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/
##########################

CSLM_DATA_FILENAME='cslm.npz'
CSLM_SOURCE_WORD_INDEX_FILENAME='source.word_index.json.gz'
CSLM_TARGET_WORD_INDEX_FILENAME='target.word_index.json.gz'

def build_sequence_data(bitext_files_context,
                        word_index_source, word_index_target,
                        source_lang='en', target_lang='fr'):
    X_train = load_bitext_sequences(
        bitext_files_context['train'].items(),
        word_index_source,
        source_lang
    )

    y_train = load_bitext_sequences(
        bitext_files_context['train'].items(),
        word_index_target,
        target_lang
    )

    X_dev = load_bitext_sequences(
        bitext_files_context['dev'].items(),
        word_index_source,
        source_lang
    )

    y_dev = load_bitext_sequences(
        bitext_files_context['dev'].items(),
        word_index_target,
        target_lang
    )

    X_test = load_bitext_sequences(
        bitext_files_context['test'].items(),
        word_index_source,
        source_lang
    )

    y_test = load_bitext_sequences(
        bitext_files_context['test'].items(),
        word_index_target,
        target_lang
    )

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_dev': X_dev,
        'y_dev': y_dev,
        'X_test': X_test,
        'y_test': y_test
    }

def load_target_word_index(cache_dir='.', overwrite=False):
    target_word_index_file = os.path.join(cache_dir, CSLM_TARGET_WORD_INDEX_FILENAME)
    bitext_files_context = load_cslm_bitext_descriptors()
    target_word_index = build_save_load_word_index(target_word_index_file,
                                                   bitext_files_context['train'].items(),
                                                   'fr',
                                                   overwrite=overwrite)
    return target_word_index

def load_source_word_index(cache_dir='.', overwrite=False):
    source_word_index_file = os.path.join(cache_dir, CSLM_SOURCE_WORD_INDEX_FILENAME)
    bitext_files_context = load_cslm_bitext_descriptors()
    source_word_index = build_save_load_word_index(source_word_index_file,
                                                   bitext_files_context['train'].items(),
                                                   'en',
                                                   overwrite=overwrite)
    return source_word_index

def _load_data(cache_dir='.', overwrite=False):
    data_file = os.path.join(cache_dir, CSLM_DATA_FILENAME)
    if not os.path.isfile(data_file) or overwrite:
        print("Data file %s does not exist, generating..." % data_file)
        bitext_files_context = load_cslm_bitext_descriptors()
        source_word_index = load_source_word_index(cache_dir=cache_dir, overwrite=overwrite)
        target_word_index = load_target_word_index(cache_dir=cache_dir, overwrite=overwrite)
        data = build_sequence_data(bitext_files_context,
                                   source_word_index,
                                   target_word_index,
                                   'en',
                                   'fr')
        np.savez_compressed(data_file, **data)
        del source_word_index
        del target_word_index
        del data
    data = np.load(data_file)
    return data

def load_data(source_vocab_size, target_vocab_size, reverse_source=False, cache_dir='.', overwrite=False):
    data = _load_data(cache_dir=cache_dir, overwrite=overwrite)
    result = []
    
    X_train = apply_vocab_size_constraint(data['X_train'], source_vocab_size)
    X_dev = apply_vocab_size_constraint(data['X_dev'], source_vocab_size)
    X_test = apply_vocab_size_constraint(data['X_test'], source_vocab_size)

    if reverse_source:
        for X in [X_train, X_dev, X_test]:
            for sequence in X:
                sequence.reverse()

    y_train = apply_vocab_size_constraint(data['y_train'], target_vocab_size)
    y_dev = apply_vocab_size_constraint(data['y_dev'], target_vocab_size)
    y_test = apply_vocab_size_constraint(data['y_test'], target_vocab_size)

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def word_counts_from_sequences(sequences):
    counts = {}
    for sequence in sequences:
        for i in sequence:
            counts[i] = counts.get(i, 0) + 1
    return counts

def load_word_counts(cache_dir=".", overwrite=False):
    result = []
    data_keys = ['X_train', 'y_train']
    data_files = [os.path.join(cache_dir, 'word_counts.{}.json.gz'.format(data_key)) for data_key in data_keys]
    data = None
    if any(not os.path.isfile(data_file) for data_file in data_files):
        data = _load_data(cache_dir=cache_dir, overwrite=overwrite)
    for data_key, data_file in zip(data_keys, data_files):
        if not os.path.isfile(data_file):
            dataset = data[data_key]
            counts = word_counts_from_sequences(dataset)
            del dataset
            with smart_open(data_file, 'w') as f:
                json.dump(counts, f)
            result.append(counts)
        else:
            with smart_open(data_file, 'r') as f:
                counts = json.load(f)
                result.append(counts)
    del data
    return result

def build_threshold_to_ids(X, y, thresholds, truncate=False):
    thresholds = sorted(thresholds)
    threshold_to_ids = {}
    for threshold in thresholds:
        threshold_to_ids[threshold] = []

    for sample_index in range(len(X)):
        en_indices, fr_indices = X[sample_index], y[sample_index]
        threshold = next((t for t in thresholds if max(len(en_indices), len(fr_indices)) <= t),
                         None)

        # add inputs and labels for this threshold
        if threshold:
            threshold_to_ids[threshold].append(sample_index)

    return threshold_to_ids
