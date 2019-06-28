"""
===============
=== Purpose ===
===============

Intercept-Sin-Cos-Holiday regression. It predicts wILI in some region on some
epiweek using ordinary regression. There are 6 covariates (7 if you count the
intercept term):
  - 4 indicator (0/1) variables for holiday weeks (50 or 51 through 01)
  - 2 timing variables: sin and cos of the epiweek

When producing retrospective predictions, great care is taken to only use
'valid' data: values that would have actually been available at the time.
However, unstable wILI is only available for recent years and for only some of
the regions (i.e. not in census regions). During training, ISCH will fall back
to stable data if unstable data is unavailable; however, during prediction,
ISCH will raise an Exception if unstable data is unavailable.

Note that the epiweek parameter represents the most recently published issue.
The returned value is a prediction for the following week.

See also:
  - arch.py: another system that generates 1-week-ahead predictions


=================
=== Changelog ===
=================

2016-04-11
  * allow predictions using invalid (stable) data
2016-04-06
  + initial version
"""

# standard library
import argparse

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
import delphi.operations.secrets as secrets
import delphi.utils.epiweek as EW
from delphi.nowcast_datasetname.util.datasetname_data_source import DatasetnameDataSource

def mutate_rows_as_if_lagged(rows, lag):
  for row in rows:
    row.update({'lag': lag})
  return rows


class ISCH:

  @staticmethod
  def dot(*Ms):
    N = Ms[0]
    for M in Ms[1:]:
      N = np.dot(N, M)
    return N

  def __init__(self, region, target):
    self.region = region
    self.target = target
    weeks = Epidata.range(199301, 202330)
    auth = secrets.api.datasetname_targets
    rx = mutate_rows_as_if_lagged(Epidata.check(Epidata.datasetname_targets(auth, self.target, self.region, weeks)), 1000000)
    self.data = {}
    self.valid = {}
    self.ew2i, self.i2ew = {}, {}
    for ew in EW.range_epiweeks(weeks['from'], weeks['to'], inclusive=True):
      # if 200916 <= ew <= 201015:
      #   continue
      i = len(self.ew2i)
      self.ew2i[ew] = i
      self.i2ew[i] = ew
    for row in rx:
      ew, observation, lag = row['epiweek'], row['value'], row['lag']
      if ew not in self.ew2i:
        continue
      i = self.ew2i[ew]
      if i not in self.data:
        self.data[i] = {}
        self.valid[i] = {'stable': False}
      lag = 'stable'
      self.data[i][lag] = observation
      self.valid[i][lag] = True
    self.weeks = sorted(list(self.data.keys()))
    for i in self.weeks:
      if 'stable' not in self.data[i]:
        continue

  def _get_features(self, ew, valid=True):
    X = np.zeros((1, 7))
    i = self.ew2i[ew]
    X[0, 0] = 1
    for holiday in range(4):
      if EW.split_epiweek(EW.add_epiweeks(ew, holiday))[1] == 1:
        X[0, 1 + holiday] = 1
    y, w = EW.split_epiweek(ew)
    N = EW.get_num_weeks(y)
    offset = np.pi * 2 * w / N
    X[0, 5] = np.sin(offset)
    X[0, 6] = np.cos(offset)
    # todo linear time trend covariate?
    return X

  def train(self, epiweek):
    if epiweek not in self.ew2i:
      raise Exception('not predicting during this period')
    i1 = self.weeks[2]
    i2 = min(self.ew2i[epiweek] - 5, self.ew2i[DatasetnameDataSource.LAST_DATA_EPIWEEK]-1)
    if i2 < i1:
      raise Exception('The available data are too "fresh"; all are being cut off for being too recent (or maybe even intersecting with the test time); at least {} additional observation(s) are needed for the training window to not cut them all off.'.format(i1-i2))
    ew1, ew2 = self.i2ew[i2], self.i2ew[i2]
    num_weeks = i2 - i1 + 1
    X, Y = np.zeros((num_weeks, 7)), np.zeros((num_weeks, 1))
    r = 0
    for i in range(i1, i2 + 1):
      X[r, :] = self._get_features(self.i2ew[i], valid=False)
      Y[r, 0] = self.data[i + 1]['stable']
      r += 1
    # rule of thumb: require num training instances >= 10x num features and >= 52
    min_training_instances = max(10*X.shape[1], 52)
    if X.shape[0] < min_training_instances:
      raise Exception('Found {} training instances, but required at least {}.'.format(X.shape[0], min_training_instances))
    # if there are (seemingly) plenty of training instances, train only on more
    # recent ones; throw out earlier training instances until there are at most
    # 50x num features of them:
    max_training_instances = 50*X.shape[1]
    X = X[-max_training_instances:,]
    Y = Y[-max_training_instances:,]
    self.model = ISCH.dot(np.linalg.inv(ISCH.dot(X.T, X)), X.T, Y)
    self.training_week = epiweek
    return (X, Y, self.model)

  def predict(self, epiweek, train=True, valid=True):
    if train:
      self.train(epiweek)
    if self.training_week > epiweek:
      raise Exception('trained on future data')
    X = self._get_features(epiweek, valid=valid)
    return float(ISCH.dot(X, self.model)[0, 0])


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('epiweek', type=int, help='most recently published epiweek (best 201030+)')
  parser.add_argument('region', type=str, help='region (state)')
  parser.add_argument('target', type=str, help='target (e.g., ov_datasetname2_per2)')
  args = parser.parse_args()

  # options
  ew1, reg, tar = args.epiweek, args.region, args.target
  ew2 = EW.add_epiweeks(ew1, 1)

  # train and predict
  print('Most recent issue: %d' % ew1)
  prediction = ISCH(reg, tar).predict(ew1, True)
  print('Predicted observation for %s in %s on %d: %.3f' % (tar, reg, ew2, prediction))
  auth = secrets.api.datasetname_targets
  res = Epidata.datasetname_targets(auth, tar, reg, ew2)
  if res['result'] == 1:
    row = res['epidata'][0]
    # issue = row['issue']
    observation = row['value']
    err = prediction - observation
    print('Actual observation as of %s: %.3f (err=%+.3f)' % ('static report', observation, err))
  else:
    print('Actual observation: unknown')

# fixme may want to be forecasting proportions or rates
# todo may want Loch Ness intercept sensor instead or in addition to this one
