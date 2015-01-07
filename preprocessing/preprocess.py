import os
import sys
import json
import socket
import pandas as pd
import preprocessing
import logging.config
from multiprocessing.pool import ThreadPool


with open(sys.argv[1]) as f:
    conf = json.load(f)
tokenizer_address = (conf['tokenizer_server']['host'], conf['tokenizer_server']['port'])
stop_words = set(conf['stop_words'])
logging.config.dictConfig(conf['logging'])
logger = logging.getLogger('preprocessing')
raw_data_path = conf['raw_data_path']
preprocessed_data_path = conf['preprocessed_data_path']


def tokenize(column_series, dataset_type):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(tokenizer_address)
    tokenizer_server = sock.makefile()

    logger.info('{}: {}'.format(dataset_type, column_series.name))
    column_series.fillna(value='', inplace=True)
    column_data = column_series.values
    for idx, text in enumerate(column_data):
        text = text.strip()
        if text == '':
            column_data[idx] = []
        else:
            tokenizer_server.write('{}\n'.format(text))
            tokenizer_server.flush()
            tokens = tokenizer_server.readline().decode('utf-8').strip()
            tokens = json.loads(tokens)
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            column_data[idx] = tokens
        if idx % 100000 == 0:
            logger.info('{}: {} {}'.format(dataset_type, column_series.name, idx))
    logger.info('{}: column {} is done!'.format(dataset_type, column_series.name))
    tokenizer_server.close()
    sock.close()

if __name__ == '__main__':
    threads = ThreadPool(processes=8)
    jobs_results = []
    preprocessed_data = {}
    for dataset_type in ['train', 'test']:
        logger.info('Start preprocessing {}'.format(dataset_type))
        raw_data = pd.read_csv(raw_data_path[dataset_type], index_col=0, low_memory=False)
        for column_name in raw_data:
            if column_name in preprocessing.text_columns:
                jobs_results.append(threads.apply_async(func=tokenize, args=(raw_data[column_name], dataset_type)))
        preprocessed_data[dataset_type] = raw_data

    for jobs_result in jobs_results:
        jobs_result.get()

    for dataset_type in ['train', 'test']:
        dir_path = os.path.dirname(preprocessed_data_path[dataset_type])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        logger.info('Saving {} ...'.format(dataset_type))
        preprocessed_data[dataset_type].to_pickle(preprocessed_data_path[dataset_type])
        logger.info('Preprocessing {} is done'.format(dataset_type))