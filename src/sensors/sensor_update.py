"""
(Nonmembers are not allowed to view this file.)

===============
=== Purpose ===
===============

Produces a signal for each flu digital surveillance source, which is then used
as a 'sensor' in the context of nowcasting through sensor fusion.

Each signal is updated over the following inclusive range of epiweeks:
  - epiweek of most recently computed signal of this type
  - last epiweek
The idea is to recompute the last stored value (just in case there were
changes to the underlying data source), and to compute all weeks up to, but
not including, the current week (because the current week is, by definition,
still ongoing).

The following signals are available:
  - ght: Google Health Trends
  - isch: Intercept-Sin-Cos-Holiday regression
  - nsnd4: DatasetnameSTAT projected/exploded to state level with NORS Dashboard, shifted 4 weeks ahead
  - nsnd7: DatasetnameSTAT projected/exploded to state level with NORS Dashboard, shifted 7 weeks ahead

See also:
  - signal_update.py
  - isch.py
"""
# standard library
import argparse
import re
import subprocess
import sys

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
from delphi.nowcast_datasetname.sensors.isch import ISCH
from delphi.nowcast_datasetname.util.datasetname_data_source import DatasetnameDataSource
from delphi.nowcast_datasetname.util.sensors_table import SensorsTable
import delphi.operations.secrets as secrets
from delphi.utils.epidate import EpiDate
import delphi.utils.epiweek as flu
from delphi.utils.geo.locations import Locations

"""
Suggestions:
1. add paramters for functions (such as fit_loch_ness) which can specify "flu" or "datasetname" or other kinds of data
2. move functions such as dot to utils.py to make the code more concise
"""


class SignalGetter:
  """Class with static methods that implement the fetching of
  different data signals. Each function returns a function that
  only takes a single argument:
  - weeks: an Epiweek range of weeks to fetch data for.
  """
  def __init__(self):
    pass

  @staticmethod
  def get_ght(location, epiweek, valid):
    loc = 'US' if location == 'nat' else location
    fetch = lambda weeks: Epidata.ght(secrets.api.ght, loc, weeks, 'datasetname')
    return fetch

  @staticmethod
  def get_datasetnamestat_norsdashboard(location, epiweek, valid, target):
    fetch = lambda weeks: Epidata.datasetnamestat_norsdashboard(secrets.api.datasetnamestat_norsdashboard, target, location, weeks)
    return fetch


class SensorFitting:
  def __init__(self):
    pass

  @staticmethod
  def fit_loch_ness(location, epiweek, name, fields, fetch, valid, target, signal_to_truth_ew_shift=0):
    # target_type is added for compatibility for other type of targets such as datasetname data

    # Helper functions
    def get_weeks(epiweek):
      ew1 = 199301
      ew2 = epiweek
      ew3 = flu.add_epiweeks(epiweek, 1)
      weeks0 = Epidata.range(ew1, ew2)
      weeks1 = Epidata.range(ew1, ew3)
      return (ew1, ew2, ew3, weeks0, weeks1)

    def extract(rows, fields, signal_to_truth_ew_shift):
      data = {}
      for row in rows:
        data[flu.add_epiweeks(row['epiweek'], signal_to_truth_ew_shift)] = [float(row[f]) for f in fields]
      return data

    def get_training_set_data(data):
      epiweeks = sorted(list(data.keys()))
      X = [data[ew]['x'] for ew in epiweeks]
      Y = [data[ew]['y'] for ew in epiweeks]
      return (epiweeks, X, Y)

    def get_training_set_datasetname(location, epiweek, signal, target, signal_to_truth_ew_shift):
      ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
      groundTruth = dict()
      auth = secrets.api.datasetname_targets
      datasetnameData = Epidata.check(Epidata.datasetname_targets(auth, target, location, weeks0))
      for row in datasetnameData:
        groundTruth[row['epiweek']] = row['value']
      data = {}
      dropped_weeks = 0
      for signal_week in signal.keys():
        ground_truth_week = flu.add_epiweeks(signal_week, signal_to_truth_ew_shift)
        # skip the week we're trying to predict
        if ground_truth_week == ew3:
          continue
        sig = signal[signal_week]
        if ground_truth_week in groundTruth:
          label = groundTruth[ground_truth_week]
        else:
          dropped_weeks += 1
          continue
        data[ground_truth_week] = {'x': sig, 'y': label}
      if dropped_weeks:
        msg = 'warning: dropped %d/%d signal weeks because ground truth / target was unavailable'
        print(msg % (dropped_weeks, len(signal)))
      epiweeks = sorted(list(data.keys()))
      X = [data[week]['x'] for week in epiweeks]
      Y = [data[week]['y'] for week in epiweeks]
      return (epiweeks, X, Y)

    def dot(*Ms):
      """ Simple function to compute the dot product
      for any number of arguments.
      """
      N = Ms[0]
      for M in Ms[1:]:
        N = np.dot(N, M)
      return N

    def get_weight(ew1, ew2):
      """ This function gives the weight between two given
      epiweeks based on a function that:
        - drops sharply over the most recent ~3 weeks
        - falls off exponentially with time
        - puts extra emphasis on the past weeks at the
          same time of year (seasonality)
        - gives no week a weight of zero
      """
      dw = flu.delta_epiweeks(ew1, ew2)
      yr = 52.2
      hl1, hl2, bw = yr, 1, 4
      a = 0.05
      # b = (np.cos(2 * np.pi * (dw / yr)) + 1) / 2
      b = np.exp(-((min(dw % yr, yr - dw % yr) / bw) ** 2))
      c = 2 ** -(dw / hl1)
      d = 1 - 2 ** -(dw / hl2)
      return (a + (1 - a) * b) * c * d

    def get_periodic_bias(epiweek):
      weeks_per_year = 52.2
      offset = flu.delta_epiweeks(200001, epiweek) % weeks_per_year
      angle = np.pi * 2 * offset / weeks_per_year
      return [np.sin(angle), np.cos(angle)]

    def apply_model(epiweek, beta, values):
      bias0 = [1.]
      if beta.shape[0] > len(values) + 1:
        # constant and periodic bias
        bias1 = get_periodic_bias(epiweek)
        obs = np.array([values + bias0 + bias1])
      else:
        # constant bias only
        obs = np.array([values + bias0])
      return float(dot(obs, beta))

    def get_model(ew2, epiweeks, X, Y):
      ne, nx1, nx2, ny = len(epiweeks), len(X), len(X[0]), len(Y)
      if ne != nx1 or nx1 != ny:
        raise Exception('length mismatch e=%d X=%d Y=%d' % (ne, nx1, ny))
      weights = np.diag([get_weight(ew1, ew2) for ew1 in epiweeks])
      X = np.array(X).reshape((nx1, nx2))
      Y = np.array(Y).reshape((ny, 1))
      bias0 = np.ones(Y.shape)
      if ne >= 26 and flu.delta_epiweeks(epiweeks[0], epiweeks[-1]) >= 52:
        # constant and periodic bias
        bias1 = np.array([get_periodic_bias(ew) for ew in epiweeks])
        X = np.hstack((X, bias0, bias1))
      else:
        # constant bias only
        X = np.hstack((X, bias0))
      XtXi = np.linalg.inv(dot(X.T, weights, X))
      XtY = dot(X.T, weights, Y)
      return np.dot(XtXi, XtY)

    if type(fields) == str:
      fields = [fields]

    ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
    rows = Epidata.check(fetch(weeks1))
    signal = extract(rows, fields, signal_to_truth_ew_shift)
    # rule of thumb: require num training instances >= 10x num features and >= 52
    min_rows = max(10*len(fields), 52)

    if ew3 not in signal:
      raise Exception('%s unavailable on %d' % (name, ew3))
    if len(signal) < min_rows:
      raise Exception('%s available less than %d weeks' % (name, min_rows))

    epiweeks, X, Y = get_training_set_datasetname(location, epiweek, signal, target, signal_to_truth_ew_shift)

    min_rows = min_rows - 1
    if len(Y) < min_rows:
      raise Exception('datasetname_targets available less than %d weeks' % (min_rows))

    model = get_model(ew3, epiweeks, X, Y)
    value = apply_model(ew3, model, signal[ew3])
    return value


class UnknownLocationException(Exception):
  """An Exception indicating that the given location is not known."""


def get_location_list(loc: str):
  """Return the list of locations described by the given string."""
  loc = loc.lower()
  if loc == 'all':
    return DatasetnameDataSource.SENSOR_LOCATIONS
  elif loc in DatasetnameDataSource.SENSOR_LOCATIONS:
    return [loc]
  else:
    raise UnknownLocationException('unknown location: %s' % str(loc))


class SensorGetter:
  """Class that implements different sensors. Some sensors
  may take in a signal to do the fitting on, others do not.
  """
  def __init__(self):
    pass
  
  @staticmethod
  def get_sensor_implementations():
    """Return a map from sensor names to sensor implementations."""
    return {
      'isch': SensorGetter.get_isch,
      'ght': SensorGetter.get_ght,
      'nsnd4': SensorGetter.get_datasetnamestat_norsdashboard_4ahead,
      'nsnd7': SensorGetter.get_datasetnamestat_norsdashboard_7ahead,
    }

  @staticmethod
  def get_isch(location, epiweek, valid, target):
    return ISCH(location, target).predict(epiweek, valid=valid)

  # sensors using the loch ness fitting

  @staticmethod
  def get_ght(location, epiweek, valid, target):
    fetch = SignalGetter.get_ght(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'ght', 'value', fetch, valid, target)

  @staticmethod
  def get_datasetnamestat_norsdashboard_4ahead(location, epiweek, valid, target):
    fetch = SignalGetter.get_datasetnamestat_norsdashboard(location, epiweek, valid, target)
    # fixme hide this signal when it would not have been available, and use appropriate release; hopefully >=4 weeks behind will be available for a monthly(?) release schedule
    return SensorFitting.fit_loch_ness(location, epiweek, 'datasetnamestat_norsdashboard_4ahead', 'value', fetch, valid, target, 4)

  @staticmethod
  def get_datasetnamestat_norsdashboard_7ahead(location, epiweek, valid, target):
    fetch = SignalGetter.get_datasetnamestat_norsdashboard(location, epiweek, valid, target)
    # fixme hide this signal when it would not have been available, and use appropriate release; hopefully >=4 weeks behind will be available for a monthly(?) release schedule
    return SensorFitting.fit_loch_ness(location, epiweek, 'datasetnamestat_norsdashboard_7ahead', 'value', fetch, valid, target, 7)


class SensorUpdate:
  """
  Produces both real-time and retrospective sensor readings for datasetname targets in the US.
  Readings (predictions of datasetname targets made using raw inputs) are stored in the Delphi
  database and are accessible via the Epidata API.
  """

  @staticmethod
  def new_instance(valid, test_mode, target):
    """
    Return a new instance under the default configuration.

    If `test_mode` is True, database changes will not be committed.

    If `valid` is True, be punctilious about hiding values that were not known
    at the time (e.g. run the model with preliminary observations only). Otherwise, be
    more lenient (e.g. fall back to final observations when preliminary data isn't
    available).
    """
    database = SensorsTable(test_mode=test_mode)
    implementations = SensorGetter.get_sensor_implementations()
    return SensorUpdate(valid, database, implementations, Epidata, target)

  def __init__(self, valid, database, implementations, epidata, target):
    self.valid = valid
    self.database = database
    self.implementations = implementations
    self.epidata = epidata
    self.target = target

  def update(self, sensors, first_week, last_week):
    """
    Compute sensor readings and store them in the database.
    """

    # most recent issue
    if last_week is None:
      # last_issue = get_most_recent_issue(self.epidata)
      # last_week = flu.add_epiweeks(last_issue, +1)
      raise Exception("last_week must be provided for now --- todo select based on current time (rather than on the ground truth data set since the ground truth here is currently static, not streaming)")

    # connect
    with self.database as database:

      # update each sensor
      for (name, loc) in sensors:

        # update each location
        for location in get_location_list(loc):

          # timing
          ew1 = first_week
          if ew1 is None:
            ew1 = database.get_most_recent_epiweek(name, location)
            if ew1 is None:
              # If an existing sensor reading wasn't found in the database and
              # no start week was given, just assume that readings should start
              # at 2010w40.
              ew1 = 201040
              print('%s-%s not found, starting at %d' % (name, location, ew1))

          args = (name, location, ew1, last_week)
          print('Updating %s-%s from %d to %d.' % args)
          for test_week in flu.range_epiweeks(ew1, last_week, inclusive=True):
            self.update_single(database, test_week, name, location)

  def update_single(self, database, test_week, name, location):
    train_week = flu.add_epiweeks(test_week, -1)
    impl = self.implementations[name]
    try:
      value = impl(location, train_week, self.valid, self.target)
      print(' %4s %5s %d -> %.3f' % (name, location, test_week, value))
    except Exception as ex:
      value = None
      print(' failed: %4s %5s %d' % (name, location, test_week), ex)
    if value is not None:
      database.insert(self.target, name, location, test_week, value)
    sys.stdout.flush()


def get_argument_parser():
  """Define command line arguments and usage."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'names',
      help=(
        'list of name-location pairs '
        '(location can be nat/hhs/cen/state or specific location labels)'))
  parser.add_argument(
      '--first',
      '-f',
      type=int,
      help='first epiweek override')
  parser.add_argument(
      '--last',
      '-l',
      type=int,
      help='last epiweek override')
  parser.add_argument(
      '--epiweek',
      '-w',
      type=int,
      help='epiweek override')
  parser.add_argument(
      '--target',
      type=str,
      default='ov_datasetname2_per2',
      help='The datasetname sensor&nowcasting target (e.g., ov_datasetname2_per2)')
  parser.add_argument(
      '--test',
      '-t',
      default=False,
      action='store_true',
      help='dry run only')
  parser.add_argument(
      '--valid',
      '-v',
      default=False,
      action='store_true',
      help='do not fall back to stable target values; require unstable target values')
  return parser


def validate_args(args):
  """Validate and return command line arguments."""

  # check epiweek specification
  first, last, week = args.first, args.last, args.epiweek
  for ew in [first, last, week]:
    if ew is not None:
      flu.check_epiweek(ew)
  if week is not None:
    if first is not None or last is not None:
      raise ValueError('`week` overrides `first` and `last`')
    first = last = week
  if first is not None and last is not None and first > last:
    raise ValueError('`first` must not be greater than `last`')

  # validate and extract name-location pairs
  pair_regex = '[^-,]+-[^-,]+'
  names_regex = '%s(,%s)*' % (pair_regex, pair_regex)
  if not re.match(names_regex, args.names):
    raise ValueError('invalid sensor specification')

  return args.names, first, last, args.valid, args.test, args.target


def parse_sensor_location_pairs(names):
  return [pair.split('-') for pair in names.split(',')]


def main(names, first, last, valid, test, target):
  """Run this script from the command line."""
  sensors = parse_sensor_location_pairs(names)
  SensorUpdate.new_instance(valid, test, target).update(sensors, first, last)


if __name__ == '__main__':
  main(*validate_args(get_argument_parser().parse_args()))
