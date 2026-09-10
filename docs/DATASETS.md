# Datasets

## Ref-YouTube-VOS

The benchmark describes 3,978 videos: 3,471 train, 202 validation, and 305 test. It reports 15,009 full-video-authored expressions (12,913 train and 2,096 over the original 507-video validation pool), 12,890 first-frame-authored expressions (10,897 train and 1,993 validation), and approximately 131k frame masks.

A first-frame expression means the annotator was shown only frame zero while writing language. It is an annotation protocol, not a request to truncate another expression's video. The first-frame-authored expression release was retired and is not reconstructed here. This repo accepts only public full-video-authored expressions and processes all listed frames.

The current public metadata exposes train, a 507-video valid pool, and a 305-video test subset. We define competition val as valid minus test, giving disjoint 202-video val and 305-video test sets. Public challenge val/test target masks are withheld; those main referents are predicted by SAM3.1 and explicitly labeled sam3.1_main_referent. Train target masks remain ground truth.

~~~text
ref-youtube-vos/
  meta_expressions/{train,valid,test}/meta_expressions.json
  archives/{train.zip,valid.zip,test_ytvos.zip}
~~~

ZIPs are read directly; mass frame extraction is unnecessary.

## ReVOS

The official 626/416 train/validation video split contains 35,074 expressions total. Public metadata used here contains:

- train: 16,941 explicit, 12,203 implicit, 108 nonexistent
- val: 3,130 explicit, 2,475 implicit, 217 nonexistent

Nonexistent descriptions are intentional negative records and are never sent to SAM. Explicit and implicit descriptions use official target sequences plus SAM3.1 contextual tracks.

~~~text
ReVOS/
  JPEGImages/<video>/<frame>.jpg
  meta_expressions_train_.json
  meta_expressions_valid_.json
  mask_dict.json
  mask_dict.sqlite
~~~

The generated SQLite file maps annotation IDs to sequences and prevents 32 workers from each parsing the large JSON mask dictionary. Dataset media and annotations are not redistributed here.
