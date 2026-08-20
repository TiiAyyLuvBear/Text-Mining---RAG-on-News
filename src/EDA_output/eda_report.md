# EDA Report

## QA Set

- QA pairs: 597
- Unique QA ids: 597
- Duplicate QA ids: 0
- Duplicate questions: 1
- Missing questions: 0
- Missing answers: 129
- Records with possible encoding/mojibake issues: 1 (0.17%)

### Question Length Stats

|        |   min |   p25 |   mean |   median |   p75 |   p95 |   max |
|:-------|------:|------:|-------:|---------:|------:|------:|------:|
| tokens |    10 |    21 |  28.58 |       25 |    32 |    56 |    96 |

### Answer Length Stats

|        |   min |   p25 |   mean |   median |   p75 |   p95 |   max |
|:-------|------:|------:|-------:|---------:|------:|------:|------:|
| tokens |     0 |     3 |  36.19 |       30 |    47 |   130 |   318 |

### QA Type Distribution

| qa_type              |   count |
|:---------------------|--------:|
| factoid              |      99 |
| event_summary        |      99 |
| cause_effect         |      85 |
| unanswerable         |      84 |
| multi_doc_comparison |      80 |
| entity_role          |      75 |
| comparison           |      38 |
| timeline             |      22 |
| claim_verification   |      15 |
