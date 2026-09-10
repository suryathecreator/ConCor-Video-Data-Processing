# Dataset layouts

The repository does not redistribute video frames, annotations, or masks. Download each dataset from its official source and point `DATA_ROOT` at the unpacked directory.

## Ref-YouTube-VOS

Supported logical splits are `train`, `val`, and `test`. Common public releases store the competition validation and test frame pools under `valid/`; the loader resolves those aliases automatically.

```text
ref-youtube-vos/
  meta_expressions/
    train/meta_expressions.json
    val/meta_expressions.json
    test/meta_expressions.json
  train/
    JPEGImages/<video_id>/<frame>.jpg
    Annotations/<video_id>/<frame>.png
    meta.json
  valid/
    JPEGImages/<video_id>/<frame>.jpg
```

Alternative flat metadata names such as `meta_expressions_train.json` are also recognized. The public train annotations provide palette masks whose integer values are object IDs. Public val/test target masks are withheld for evaluation, so the pipeline uses SAM3.1 for the main referent on those splits and records `target_source=sam3.1_multiplex`.

Only the full-video-authored language subset is currently public. `--ref-mode first_frame` is a compute/view choice that retains only the first frame from those expressions. `--ref-mode full_video` processes every listed frame.

Official project page: <https://youtube-vos.org/dataset/rvos/>

## ReVOS

Supported logical splits are `train` and `val`; `val` resolves the upstream `valid` spelling.

```text
ReVOS/
  JPEGImages/<video_id>/<frame>.jpg
  # JPEGImages.zip may be used instead of the unpacked directory.
  mask_dict.json
  meta_expressions_train_.json
  meta_expressions_valid_.json
```

Choose any subset of `implicit`, `explicit`, and `nonexistent`. Explicit and implicit expressions retain their official target RLE tracklets and add SAM3.1 context. Nonexistent-object expressions are valid negatives: the text is preserved, model segmentation is skipped, and the record contains zero tracklets.

Official API/download page: <https://github.com/cilinyan/ReVOS-api>

## Selection

Selection is stable for a fixed dataset release, seed, and configuration. Without a limit, expressions are processed in sorted `(video_id, expression_id)` order. With `LIMIT` or `LIMIT_PER_CATEGORY`, candidates are seeded and sampled. Set `ONE_EXPRESSION_PER_VIDEO=1` when a sample should contain at most one expression from each video.

The generated `worklist.json` freezes every selected identifier, frame sequence, exact extraction decision, and input path. Resume uses that file instead of sampling again.
