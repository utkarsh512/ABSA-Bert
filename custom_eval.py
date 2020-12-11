# Evaluating performance of a model after its training
# As we are using a different approach to generate training and testing datasets
# hence we can't use 'evaluation.py'
#
# Evaluates a model on various metrics
# How to use:
# Suppose you want to evaluate performance of 'QA_M'. Let 'result/QA_M/test_ep_6.txt'
# be the directory of results generated for last epoch then write (in linux terminal):
#
# $ python custom_eval.py --task_name QA_M --path result/QA_M/test_ep_6.txt
#
# @author utkarsh512

import argparse
import numpy as np
import pandas as pd 
from sklearn import metrics

def get_y_true(task_name):
  assert task_name in ['QA_M', 'QA_B', 'NLI_M', 'NLI_B'], "error!"
  data_dir = 'data/sentihood/bert-pair/test_'
  
  y_true = []
  path = data_dir + task_name + '.tsv'
  df = pd.read_csv(path, sep='\t')

  if task_name in ['QA_M', 'NLI_M']:
    for i in range(len(df)):
      label = df['label'][i]
      assert label in ['None', 'Positive', 'Negative'], "error!"
      n = None
      if label == 'None':
        n = 0
      elif label == 'Positive':
        n = 1
      else:
        n = 2
      y_true.append(n)
  
  else:
    for i in range(len(df)):
      label = df['label'][i]
      y_true.append(label)
  
  return y_true


def get_y_pred(path):
  y_pred = []
  score = []
  with open(path, 'r', encoding='utf-8') as f:
    S = f.readlines()
    for s in S:
      s = s.strip().split()
      y_pred.append(int(s[0]))
      cur_score = []
      for i in range(1, len(s)):
        cur_score.append(float(s[i]))
      score.append(cur_score)
  return y_pred, score  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_name', default=None, type=str, required=True, choices=['QA_M', 'QA_B', 'NLI_M', 'NLI_B'], help='The name of the task to evaluate')
  parser.add_argument('--path', default=None, type=str, required=True, help='Directory of predicted values')
  args = parser.parse_args()
  y_true = get_y_true(args.task_name)
  y_pred, score = get_y_pred(args.path)

  # Normalizing score for ROC-AUC score
  if args.task_name[-1] == 'M':
    for i in range(len(score)):
      s = 0
      for j in range(len(score[0])):
        s += score[i][j]
      for j in range(len(score[0])):
        score[i][j] /= s
    score = np.array([np.array(xi) for xi in score])
  else:
    new_score = []
    for i in range(len(score)):
      neg, pos = score[i][0], score[i][1]
      tot = neg + pos
      pos /= tot
      new_score.append(pos)
    score = np.array(new_score)
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  score = np.array([np.array(xi) for xi in score])
  result = dict()
  result['accuracy'] = metrics.accuracy_score(y_true, y_pred)
  result['precision_macro'] = metrics.precision_score(y_true, y_pred, average='macro')
  result['precision_weighted'] = metrics.precision_score(y_true, y_pred, average='weighted')
  result['recall_macro'] = metrics.recall_score(y_true, y_pred, average='macro')
  result['recall_weighted'] = metrics.recall_score(y_true, y_pred, average='weighted')
  result['f1_macro'] = metrics.f1_score(y_true, y_pred, average='macro')
  result['f1_weighted'] = metrics.f1_score(y_true, y_pred, average='weighted')
  if args.task_name[-1] == 'M':
    result['auc_score_macro'] = metrics.roc_auc_score(y_true, score, average='macro', multi_class='ovo')
    result['auc_score_weighted'] = metrics.roc_auc_score(y_true, score, average='weighted', multi_class='ovo')
  else:
    result['auc_score_macro'] = metrics.roc_auc_score(y_true, score)
    result['auc_score_weighted'] = metrics.roc_auc_score(y_true, score)
  print('Results')
  print('-' * 20)
  for key in result.keys():
    print('{0} = {1}'.format(key, result[key]))

if __name__ == '__main__':
  main()
