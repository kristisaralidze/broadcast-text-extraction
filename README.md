# Broadcast Text Extraction

Extracting structured information from Portuguese election broadcast media: segmenting
continuous newscasts into individual stories, attributing debate speech to candidates,
and auditing the reliability of automatic transcripts before anything downstream depends
on them.

Corpus: 14 full `Telejornal` broadcasts from RTP and TVI, plus the televised
presidential debates, recorded around the 2026 Portuguese presidential election.

---

## The problem

A newscast arrives as one continuous recording. To analyse coverage you first need to
know where each story starts and ends, and that boundary information is not given
anywhere in the source.

The obvious approach was to segment on the speech transcripts: embed each utterance and
cut where the topic shifts. **That did not work.** Anchor hand-offs, studio chatter and
advertising breaks produced embedding shifts as large as real topic changes, so the
boundaries were not separable.

The approach that did work was to ignore what was said and read what was shown. RTP and
TVI both caption every story with an on-screen lower-third headline. Running OCR over
sampled frames and tracking when that headline changes gives a boundary signal that is
independent of the audio entirely.

## What the notebooks do

| Notebook | Contents |
|---|---|
| `01_speech_eda_and_quality_audit` | Schema inventory across 14 broadcasts, a nine-check quality audit, manual review of flagged segments, advertisement and promo detection, partial-capture detection, linguistic profiling, embedding geometry |
| `02_story_segmentation` | OCR lower-third parsing, headline normalisation and smoothing, scene boundary detection, story-level aggregation, thematic grouping, cross-modal validation |
| `03_ocr_exploration` | Exploratory analysis of the raw OCR output and its failure modes |
| `04_newscast_debate_correlation` | Relating newscast story coverage to televised debate content |
| `05_cross_channel_story_similarity` | Comparing how the two channels covered the same stories |

Around 3,500 lines of Python across the five newscast notebooks, plus the debate
speech processing in `debates/`.

## Debate speech processing

`debates/`

The same corpus includes the televised debates, where the problem is different. There
are no on-screen headlines to segment on, and the useful unit is not a story but a
stretch of speech attributable to one candidate.

| File | Contents |
|---|---|
| `processing/get_audio_and_speech_data.py` | Assembles aligned audio and transcript records |
| `processing/get_cleaned_audio_data.py` | Cleaning, with KMeans over acoustic features to separate speech conditions |
| `processing/get_segment_information.py` | Builds one dataset of filtered segments, each labelled by candidate |
| `01_speech_information.ipynb` | Topic identification over the transcripts |
| `02_audio_and_speech_dataframe.ipynb` | Assembles the combined analysis frame |

`get_segment_information.py` carries its own caveat in a comment at the top, noting that
this is the simpler version of the mapping. Attributing a segment to a speaker is where
most of the error enters, and the code says so rather than presenting the labels as
settled.

## Treating the text as unreliable by default

Automatic transcripts and OCR are both noisy, and the useful question is not whether the
output is clean but which parts of it can carry an inference.

**Nine segment-level quality checks** run over every speech segment, each stored as its
own boolean column rather than collapsed into a single pass or fail. Among them a
repeated-5-gram check for transcription loops, a foreign-language share check, and
duration outliers. The flags are kept alongside the data so any later analysis can decide
its own tolerance.

**Flagged segments were read individually.** The repeated-5-gram check raised 8 segments;
each was inspected and labelled with a verdict (`whisper_loop`, `off_topic_promo`) rather
than dropped on the strength of the heuristic alone.

**Partial recordings were detected and excluded.** Some captures begin mid-broadcast.
A full `Telejornal` runs to a known duration, so short ones are identifiable and are held
out of any analysis that assumes a complete episode.

**OCR output was repaired before use.** Known character fusions are corrected, headlines
are normalised, and the headline track is smoothed so that momentary OCR dropouts do not
fragment one story into several. Spurious short runs are filtered out.

## Validation

Section 5 of `02_story_segmentation` checks the OCR-derived boundaries against the speech
embeddings. The two signals are independent: one comes from pixels, the other from audio.
Where they agree, the boundary is credible; where they disagree, the disagreement is
reported rather than smoothed away.

This is cross-modal agreement between two automatic signals. It is not a comparison
against human-annotated boundaries, and it is not presented as one.

## Reproducibility

Seeds are fixed (`42`) and library versions are printed at the top of each notebook.
Outputs are committed, so the full analysis can be read without access to the source data.

## Data availability

**The source data is not in this repository.** It consists of OCR and speech transcripts
derived from RTP and TVI broadcasts, which are copyrighted works recorded for academic
use. Redistributing them is not ours to do.

The notebooks expect two files per broadcast under `data/`:

```
data/
  Telejornal_<channel>_<date>_ocr.pkl      one row per sampled frame, with text boxes
  Telejornal_<channel>_<date>_speech.pkl   one row per speech segment, with embeddings
```

Speech embeddings were produced with **EuroLLM**; linguistic processing uses spaCy
`pt_core_news_md` (tokeniser, lemmatiser, NER).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_md
jupyter lab notebooks/
```

## Authors and contribution

This was a four-person university project (Processing Big Data, Instituto Superior
Técnico) covering both newscasts and televised debates.

**This repository contains the text and speech work, which is mine** (Kristine
Saralidze): the OCR parsing and segmentation pipeline, the speech transcript quality
audit, the cross-channel story analysis, and the debate speech processing.

The video and visual analysis was carried out by **Diego Soler**, **Rediet Mulugeta**
and **Truls Saether**, and is not included here.

## Licence

Copyright (c) 2026 Kristine Saralidze. All rights reserved for the code in this
repository. Broadcast content referenced in the analysis remains the property of its
respective broadcasters.
