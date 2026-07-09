# EDA Report

## Article Corpus

- Articles: 10,073
- Unique article ids: 10,073
- Duplicate article ids: 0
- Missing title: 0
- Missing description: 0
- Missing content: 0
- Records with possible encoding/mojibake issues: 131 (1.3%)
- Categories: 13

### Content Length Stats

|       |   min |   p25 |    mean |   median |   p75 |    p95 |    max |
|:------|------:|------:|--------:|---------:|------:|-------:|-------:|
| chars |   221 |  1953 | 3151.61 |     2786 |  3912 | 6267.2 | 163370 |

|        |   min |   p25 |   mean |   median |   p75 |    p95 |   max |
|:-------|------:|------:|-------:|---------:|------:|-------:|------:|
| tokens |    50 |   423 | 685.76 |      606 |   852 | 1370.4 | 36122 |

### Top Categories

| category           |   count |
|:-------------------|--------:|
| Giải trí           |    1580 |
| Kinh doanh         |    1276 |
| Thể thao           |    1184 |
| Thời sự            |    1010 |
| Giáo dục           |     856 |
| Khoa học công nghệ |     752 |
| Xe                 |     741 |
| Đời sống           |     728 |
| Sức khỏe           |     634 |
| Thế giới           |     596 |
| Du lịch            |     329 |
| Bất động sản       |     268 |
| Pháp luật          |     119 |

## QA Set

- QA pairs: 152
- Unique QA ids: 152
- Duplicate QA ids: 0
- Duplicate questions: 0
- Missing questions: 0
- Missing answers: 39
- Records with possible encoding/mojibake issues: 0 (0.0%)
- Referenced article ids: 46
- Referenced article ids found in article file: 20
- Missing article references: 52

### Question Length Stats

|        |   min |   p25 |   mean |   median |   p75 |   p95 |   max |
|:-------|------:|------:|-------:|---------:|------:|------:|------:|
| tokens |    12 |    19 |  32.11 |       24 |    45 |  66.9 |    81 |

### Answer Length Stats

|        |   min |   p25 |   mean |   median |   p75 |   p95 |   max |
|:-------|------:|------:|-------:|---------:|------:|------:|------:|
| tokens |     0 |     0 |  48.08 |     43.5 |  69.5 | 137.9 |   192 |

### QA Type Distribution

| qa_type              |   count |
|:---------------------|--------:|
| multi_doc_comparison |      42 |
| cause_effect         |      21 |
| factoid              |      20 |
| event_summary        |      20 |
| unanswerable         |      15 |
| entity_role          |      14 |
| timeline             |      10 |
| comparison           |       5 |
| claim_verification   |       5 |
