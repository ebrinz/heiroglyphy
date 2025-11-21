---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- egy
- de
- en
license:
- cc-by-sa-4.0
multilinguality:
- multilingual
size_categories:
- 100K<n<1M
source_datasets:
- extended|wikipedia
task_categories:
- translation
task_ids: []
pretty_name: BBAW, Thesaurus Linguae Aegyptiae, Ancient Egyptian (2018)
dataset_info:
  features:
  - name: transcription
    dtype: string
  - name: translation
    dtype: string
  - name: hieroglyphs
    dtype: string
  splits:
  - name: train
    num_bytes: 18533905
    num_examples: 100736
  download_size: 9746860
  dataset_size: 18533905
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for "bbaw_egyptian"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [https://edoc.bbaw.de/frontdoor/index/index/docId/2919](https://edoc.bbaw.de/frontdoor/index/index/docId/2919)
- **Repository:** [Github](https://phiwi.github.io/all.json)
- **Paper:** [Multi-Task Modeling of Phonographic Languages: Translating Middle Egyptian Hieroglyph](https://zenodo.org/record/3524924)
- **Point of Contact:** [Philipp Wiesenbach](https://www.cl.uni-heidelberg.de/~wiesenbach/index.html)
- **Size of downloaded dataset files:** 35.65 MB


### Dataset Summary

This dataset comprises parallel sentences of hieroglyphic encodings, transcription and translation as used in the paper [Multi-Task Modeling of Phonographic Languages: Translating Middle Egyptian Hieroglyph](https://zenodo.org/record/3524924). The data triples are extracted from the [digital corpus of Egyptian texts](https://edoc.bbaw.de/frontdoor/index/index/docId/2919) compiled by the project "Strukturen und Transformationen des Wortschatzes der ägyptischen Sprache".

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

The dataset consists of parallel triples of
- `hieroglyphs`: [Encoding of the hieroglyphs with the [Gardiner's sign list](https://en.wikipedia.org/wiki/Gardiner%27s_sign_list)
- `transcription`: Transliteration of the above mentioned hieroglyphs with a [transliteration scheme](https://en.wikipedia.org/wiki/Transliteration_of_Ancient_Egyptian)
- `translation`: Translation in mostly German language (with some English mixed in)

## Dataset Structure

The dataset is not divided into 'train', 'dev' and 'test' splits as it was not built for competitive purposes and we encourage all scientists to use individual partitioning schemes to suit their needs (due to the low resource setting it might be advisable to use cross validation anyway). The only available split 'all' therefore comprises the full 100,708 translation triples, 35,503 of which possess hieroglyphic encodings (the remaining 65,205 triples have empty `hieroglyph` entries).

### Data Instances

An example of a data triple looks the following way:

```
{
    "transcription": "n rḏi̯(.w) gꜣ =j r dbḥ.t m pr-ḥḏ",
    "translation": "I was not let to suffer lack in the treasury with respect to what was needed;",
    "hieroglyphs": "D35 D21 -D37 G1&W11 -V32B A1 D21 D46 -D58 *V28 -F18 *X1 -A2 G17 [? *O2 *?]"
}

```

*Important*: Only about a third of the instance actually cover hieroglyphic encodings (the rest is the empty string `""`) as the leftover encodings have not yet been incorporated into the BBAW's project database.

### Data Fields

#### plain_text
- `transcription`: a `string` feature.
- `translation`: a `string` feature.
- `hieroglyphs`: a `string` feature.


### Data Splits

|   name   |all|
|----------|----:|
|plain_text|100708|

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

The data source comes from the project "Strukturen und Transformationen des Wortschatzes der ägyptischen Sprache" which is compiling an extensively annotated digital corpus of Egyptian texts. Their [publication](https://edoc.bbaw.de/frontdoor/index/index/docId/2919) comprises an excerpt of the internal database's contents.

#### Who are the source language producers?

[More Information Needed]


### Annotations

#### Annotation process

The corpus has not been preprocessed as we encourage every scientist to prepare the corpus to their desired needs. This means, that all textcritic symbols are still included in the transliteration and translation. This concerns the following annotations:

- `()`: defective
- `[]`: lost
- `{}`: surplus
- `〈〉`: omitted
- `⸢⸣`: damaged
- `⸮?`: unclear
- `{{}}`: erasure
- `(())`: above
- `[[]]`: overstrike
- `〈〈〉〉`: haplography

Their exists a similar sign list for the annotation of the hieroglyphic encoding. If you wish access to this list, please get in contact with the author.


#### Who are the annotators?

AV Altägyptisches Wörterbuch (https://www.bbaw.de/forschung/altaegyptisches-woerterbuch), AV Wortschatz der ägyptischen Sprache (https://www.bbaw.de/en/research/vocabulary-of-the-egyptian-language, https://aaew.bbaw.de); 
Burkhard Backes, Susanne Beck, Anke Blöbaum, Angela Böhme, Marc Brose, Adelheid Burkhardt, Roberto A. Díaz Hernández, Peter Dils, Roland Enmarch, Frank Feder, Heinz Felber, Silke Grallert, Stefan Grunert, Ingelore Hafemann, Anne Herzberg, John M. Iskander, Ines Köhler, Maxim Kupreyev, Renata Landgrafova, Verena Lepper, Lutz Popko, Alexander Schütze, Simon Schweitzer, Stephan Seidlmayer, Gunnar Sperveslage, Susanne Töpfer, Doris Topmann, Anja Weber

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

CC BY-SA 4.0 Deed Attribution-ShareAlike 4.0 International https://creativecommons.org/licenses/by-sa/4.0/ 

### Citation Information
Source corpus:
```
@misc{BerlinBrandenburgischeAkademiederWissenschaften2018,
 editor = {{Berlin-Brandenburgische Akademie der Wissenschaften} and {Sächsische Akademie der Wissenschaften zu Leipzig} and Richter, Tonio Sebastian and Hafemann, Ingelore and Hans-Werner Fischer-Elfert and Peter Dils},
 year = {2018},
 title = {Teilauszug der Datenbank des Vorhabens {\dq}Strukturen und Transformationen des Wortschatzes der {\"a}gyptischen Sprache{\dq} vom Januar 2018},
 url = {https://nbn-resolving.org/urn:nbn:de:kobv:b4-opus4-29190},
 keywords = {493;932;{\"A}gyptische Sprache;Korpus},
 abstract = {The research project {\dq}Strukturen und Transformationen des Wortschatzes der {\{\dq}a}gyptischen Sprache{\dq} at the Berlin-Brandenburgische Akademie der Wissenschaften compiles an extensively annotated digital corpus of Egyptian texts. This publication comprises an excerpt of the internal database's contents. Its JSON encoded entries require approximately 800 MB of disk space after decompression.},
 location = {Berlin},
 organization = {{Berlin-Brandenburgische Akademie der Wissenschaften} and {Sächsische Akademie der Wissenschaften zu Leipzig}},
 subtitle = {Database snapshot of project {\dq}Strukturen und Transformationen des Wortschatzes der {\"a}gyptischen Sprache{\dq} (excerpt from January 2018)}
}
```

Translation paper:
```
@article{wiesenbach19,
  title = {Multi-Task Modeling of Phonographic Languages: Translating Middle Egyptian Hieroglyphs},
  author = {Wiesenbach, Philipp and Riezler, Stefan},
  journal = {Proceedings of the International Workshop on Spoken Language Translation},
  journal-abbrev = {IWSLT},
  year = {2019},
  url = {https://www.cl.uni-heidelberg.de/statnlpgroup/publications/IWSLT2019_v2.pdf}
}
```

### Contributions

Thanks to [@phiwi](https://github.com/phiwi) for adding this dataset.