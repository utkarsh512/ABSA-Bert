import argparse
import pandas as pd
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', default=0.7, type=float, required=False, help='Ratio of train set wrt original dataset')
  parser.add_argument('--dev', default=0.1, type=float, required=False, help='Ratio of dev set wrt original dataset')
  args = parser.parse_args()

  path = 'data/sentihood/bert-pair/'
  models = ['QA_M', 'NLI_M', 'QA_B', 'NLI_B']

  TRAIN = args.train
  DEV = args.dev
  
  for model in models:
    train_path = path + 'train_' + model + '.tsv'
    dev_path = path + 'dev_' + model + '.tsv'
    test_path = path + 'test_' + model + '.tsv'
    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    os.system('rm -fv {0} {1} {2}'.format(train_path, dev_path, test_path))
    df = pd.concat([test_df, dev_df, train_df])
    df = df.reset_index(drop=True)
    LEN = len(df)
    Z = 4 if model[-1] == 'M' else 12
    LEN = LEN // Z
    A = int(TRAIN * LEN) * Z
    B = int((TRAIN + DEV) * LEN) * Z
    train_df = df[: A]
    dev_df = df[A : B]
    test_df = df[B :]
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print('Model {4} - Length of datasets: Whole {0}, train {1}, dev {2} and test {3}'.format(len(df), len(train_df), len(dev_df), len(test_df), model))
    train_df.to_csv(train_path, sep='\t', index=False)
    dev_df.to_csv(dev_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)
