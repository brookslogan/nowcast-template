CREATE TABLE `datasetname_targets`(
  `id`            INT(11)      NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `target`        VARCHAR(32)  NOT NULL                           ,
  `epiweek`       INT(11)      NOT NULL                           ,
  `location`      VARCHAR(12)  NULL                               ,
  -- todo: release/issue?
  `value`         FLOAT        NOT NULL                           ,
  UNIQUE KEY `entry` (`target`, `epiweek`, `location`),
  KEY `target` (`target`),
  KEY `epiweek` (`epiweek`),
  KEY `location` (`location`)
);
