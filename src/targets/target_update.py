"""
===============
=== Purpose ===
===============

Prepare sensor&nowcast targets from data sources for ground truth, and record
in the database (see CREATE TABLE IF NOT EXISTS statement for associated table
`datasetname_targets` below). This could be as simple as copying a count or
calculating a proportion from two fields in the data source. However,
currently, the only target is a little more complicated:
- target 'ov_datasetname2_per2': a modification of the percentage of total office
  visits with a datasetname-broad code anywhere-listed
  + a small proportion (0.01) of the numerator and denominator visits are
    uniformly smeared across all recorded geographical units to provide a less
    noisy estimate for low-denominator observations (due to low population or
    low network coverage at the observation time); and
  + a single pseudocount is uniformly smeared across the denominators for all
    recorded geographical units
  + "datasetname2": refers to "datasetname broad" spec
  + "per2": refers to modified percentage
"""

# third party
import pandas as pd
import mysql.connector

# first party
from delphi.epidata.client.delphi_epidata import Epidata
import delphi.operations.secrets as secrets
from delphi.utils.epidate import EpiDate

def main():
  print('Reading in data and calculating targets...')
  # Load entire data set, adding 1 location at a time:
  # - first, get all locations from metadata:
  datasetname_info = Epidata.check(Epidata.meta_datasetname(secrets.api.datasetname))
  locations = [row['location'] for row in datasetname_info['locations']]
  # - now iterate through the locations and add the data:
  datasetname_df = None
  for location in locations:
    location_data = Epidata.check(Epidata.datasetname(secrets.api.datasetname, location, Epidata.range(123412, EpiDate.today().get_ew())))
    location_df = pd.DataFrame(location_data)
    if datasetname_df is None:
      datasetname_df = location_df
    else:
      datasetname_df = pd.concat([datasetname_df, location_df], copy=False)
  # Prepare the target values:
  target_df = (
    datasetname_df
    # IMPL: transform data into target values here, e.g., preparing rate estimates from counts, etc.
    .assign(
      target = 'datasetname_rate', # name/label for the target
      value = lambda df: df['rate']
    )
    # lowercase location to match delphi-epidata codes:
    .assign(location = lambda df: df['location'].str.lower())
    # restrict to desired columns:
    [['target','epiweek','location','value']]
  )
  print('Target values calculated.  Recording in database...')
  # Add to database:
  (u, p) = secrets.db.epi
  cnx = mysql.connector.connect(user=u, password=p, database='epidata')
  try:
    cursor = cnx.cursor()
    ## set up dest table if doesn't exist:
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS `datasetname_targets`(
        `id`            INT(11)      NOT NULL PRIMARY KEY AUTO_INCREMENT,
        `target`        VARCHAR(32)  NOT NULL                           ,
        `epiweek`       INT(11)      NOT NULL                           ,
        `location`      VARCHAR(12)  NULL                               ,
        `value`         FLOAT        NOT NULL                           ,
        UNIQUE KEY `entry` (`target`, `epiweek`, `location`),
        KEY `target` (`target`),
        KEY `epiweek` (`epiweek`),
        KEY `location` (`location`)

        -- todo: release/issue?

      );
    ''')
    cursor.executemany('''
      INSERT INTO `datasetname_targets` (`target`, `epiweek`, `location`, `value`)
      VALUES (%s, %s, %s, %s)
      ON DUPLICATE KEY UPDATE `target`=VALUES(`target`), `epiweek`=VALUES(`epiweek`), `location`=VALUES(`location`), `value`=VALUES(`value`)
    ''', [(target, epiweek, location, value) for
          (target, epiweek, location, value) in target_df[['target','epiweek','location','value']].itertuples(index=False, name=None)
    ])
    cnx.commit()
    print('Successfully recorded target data.')
  finally:
    cnx.close()

if __name__ == '__main__':
  main()

# todo command-line interface like sensor_update




