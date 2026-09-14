# Log Catalog and Modeling Caveats

All eight files are raw XES 1.0 logs. Counts below were verified directly from
the packaged files with `tools/inspect_xes.py`.

## BPI Challenge 2013 - Volvo IT VINST

| Log | Cases | Events | `concept:name` values | Official activity classifier values | Mean / max events per case | Time coverage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Closed Problems | 1,487 | 6,660 | 4 | 7 | 4.479 / 35 | 2006-01-11 to 2012-05-31 |
| Incidents | 7,554 | 65,533 | 4 | 13 | 8.675 / 123 | 2010-03-31 to 2012-05-23 |
| Open Problems | 819 | 2,351 | 3 | 5 | 2.871 / 22 | 2006-11-07 to 2012-06-15 |

The official activity classifier is the pair
`concept:name + lifecycle:transition`, not `concept:name` alone. This explains
why the benchmark paper reports 7 and 13 activities for Closed Problems and
Incidents even though there are only four top-level status names.

The three logs share the following event-level fields:

- `concept:name`
- `lifecycle:transition`
- `time:timestamp`
- `impact`
- `org:group`
- `org:resource`
- `org:role` (partially missing)
- `organization country`
- `organization involved`
- `product`
- `resource country`

Important exceptions:

- Open Problems contains unfinished cases. It is useful for inference under
  censoring or out-of-domain evaluation, but final duration, remaining time, and
  complete-suffix targets are not observed.
- Open Problems misspells `organization country` as `oranization country` and
  does not include `organization involved`.
- The 2013 logs contain timestamps spanning years and highly heavy-tailed case
  durations. Report time units explicitly.
- Resource, group, and product identifiers are high-cardinality anonymized
  categories; fit their vocabularies on training data only.

## BPI Challenge 2020 - TU/e reimbursement processes

| Log | Cases | Events | Activities | Mean / max events per case | Observed timestamp range |
| --- | ---: | ---: | ---: | ---: | --- |
| Domestic Declarations | 10,500 | 56,437 | 17 | 5.375 / 24 | 2017-01-09 to 2019-06-17 |
| International Declarations | 6,449 | 72,151 | 34 | 11.188 / 27 | 2016-10-05 to 2020-05-10 |
| Prepaid Travel Cost | 2,099 | 18,246 | 29 | 8.693 / 21 | 2017-01-09 to 2019-02-21 |
| Request For Payment | 6,886 | 36,796 | 19 | 5.344 / 20 | 2017-01-09 to 2019-08-08 |
| Travel Permit (`PermitLog`) | 7,065 | 86,581 | 51 | 12.255 / 90 | 2016-10-05 to 2021-09-01 |

Every BPI 2020 event has:

- `id`
- `concept:name`
- `time:timestamp`
- `org:resource`
- `org:role`

Case-level fields vary substantially by log. Inspect them with
`tools/inspect_xes.py`; `PermitLog` in particular contains repeated indexed
fields for related declarations and requests.

Important exceptions:

- The five logs are related views, not five independent samples. `PermitLog`
  explicitly includes events associated with permits, prepaid requests, and
  declarations. A random cross-log split can therefore leak the same underlying
  business activity across train and test.
- The challenge documentation says the source records cover the 2017 pilot and
  the 2018 full rollout. Some logged `Start trip`/`End trip` timestamps extend
  beyond that period because they represent estimated travel dates.
- The 2017 process differs from the 2018 process, creating a natural temporal
  distribution shift.
- Amounts and identifiers are anonymized. Amounts are perturbed but remain
  approximately meaningful in aggregate and within related permits.
- Activities combine an object, status, and acting role in a single string, for
  example `Declaration APPROVED by ADMINISTRATION`. Whether to decompose that
  field is a modeling decision, not part of this bundle.

## Split hygiene

- Never split individual events from one case across train and test.
- Fit categorical vocabularies, scalers, imputers, and feature selection on the
  training partition only.
- State whether an end-of-case marker is added.
- For chronological evaluation, define the ordering key and how overlapping
  cases are handled.
- Treat Open Problems as right-censored unless a different assumption is
  explicitly justified.
- If evaluating transfer among BPI 2020 logs, audit cross-log permit and
  declaration identifiers before claiming zero-shot generalization.
