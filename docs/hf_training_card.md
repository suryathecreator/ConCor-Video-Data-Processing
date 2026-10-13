# ConCor Video Data (Filtered)

Ref-YouTube-VOS and ReVOS annotations linking exact caption spans to full instance-mask tracklets in both directions (**BCC: Bidirectional Concept Correspondence**). GPT refinement addresses the main observed failure modes: incorrect spaCy entity links, irrelevant or excessive context masks, unsupported descriptions, and broken correspondences. Automated review improves these annotations; it does not certify every mask or caption as correct.

## Training files and evaluation

| File / split | Rows | Meaning |
| --- | ---: | --- |
| [train_filtered.parquet](train_filtered.parquet) | ${train_filtered_rows} | GPT-refined training, **only `disposition == "complete_bcc"`**. Recommended for complete positive BCC supervision. |
| [train_all.parquet](train_all.parquet) | ${train_all_rows} | The same GPT pipeline, without the complete-BCC-only filter. Includes 60 accepted nonexistent-target examples. |
| [train_old/verification.parquet](train_old/verification.parquet) | ${train_old_rows} | **Original training data, before GPT filtering**, moved from `train/` without changing its bytes. All 22 original columns remain together. |
| [eval/unfiltered/verification.parquet](eval/unfiltered/verification.parquet) | 7,918 | Original evaluation annotations; not GPT-refined. |
| [eval/filtered/verified.parquet](eval/filtered/verified.parquet) | 1,150 | Existing manually reviewed evaluation subset; overlaps the unfiltered evaluation data. |

`disposition` is retained in both new files; `complete_bcc` is its value, not a separate column. `train_all` contains `complete_bcc` and `negative_unsegmentable` rows. These are alternate training views: **do not concatenate them**. The original training manifest is preserved in [train_old/manifest.json](train_old/manifest.json); evaluation files and review decisions are unchanged.

```python
from datasets import load_dataset
train = load_dataset("suryadv/concor-video-data-filtered", "gpt_refined",
                     split="train_filtered")  # or "train_all"
old = load_dataset("suryadv/concor-video-data-filtered", "original",
                   split="train_old")  # also eval_unfiltered / eval_filtered
```

## Pipeline and prompt sources

Generation uses the [ConCor Video Data Processing repository](https://github.com/suryathecreator/ConCor-Video-Data-Processing): preserve available public target masks, extract context entities, segment/track extra entities with **SAM 3.1 Object Multiplex**, build both BCC directions, and upload annotations to HF. Refinement then reads the downloaded HF training annotations with local source media.

Both refinement passes use **GPT-6 Luna, medium reasoning**, through the Batch API, across all **42,165 training expressions / 4,097 videos**. No evaluation rows or overlapping manually reviewed subset were added. Prompts adapt the original [ConCor BCC rules](https://github.com/suryathecreator/ConCor-Training-Data-Generation) and [Weiyang Huang's VIPOSeg](https://github.com/Weiyang-Huang/VIPOSeg-Grounded-Caption) visual presentation, temporal grounding, and language rules.

1. **Cheap structural filtering:** reject malformed/missing/empty positive target tracks and pathological examples above 64 distinct nonempty tracks; remove empty context tracks and exact duplicates. Reject the entire expression if one entity mention/group has **more than 10 distinct nonempty tracks**, counting the union across groups. Exactly 10 is allowed; this is not a scene-wide object limit.
2. **[Text-only NLP cleanup](https://github.com/suryathecreator/ConCor-Video-Data-Processing/blob/main/prompts/nlp_cleanup.md):** preserve the exact sentence and intended target while repairing noun-phrase boundaries/coreference and removing spaCy links to verbs such as “facing,” abstract directions, punctuation, and generic taxonomy glosses. This pass does not select visual instances.
3. **[Visual caption/BCC review](https://github.com/suryathecreator/ConCor-Video-Data-Processing/blob/main/prompts/video_refinement.md):** sample **12 evenly spaced frame indices including endpoints, plus each supplied track's first visible frame**. Each original frame sits beside its numbered, colored mask overlay (maximum height 540 px, JPEG quality 88). This is index sampling, not FPS sampling or exhaustive inspection; there are no extra crops or visual passes. Share frames across up to **12 expressions from the same dataset/split/video**; share IDs only for exactly identical complete tracks, preserve each expression's permitted IDs, and split groups above 24 distinct tracks.
4. **Keep, rewrite, or reject in one visual call:** preserve the intended target and supported temporal meaning; remove irrelevant/wrong context and unsupported unsegmented background phrases. Use natural present-tense language without invented events or camera narration. Ground concrete mentions and coreferences, not taxonomy definitions—e.g. ground “Which animals,” not the explanatory “(penguins).” Reject irreparable targets/masks. Accept intentional nonexistent targets by default, unchanged and unlinked; reject them only when separately mentioned concrete context lacks compatible masks (or fails structural/link limits).
5. **Local validation and export:** constrain allowed IDs, validate tags and exact Unicode character spans, preserve positive targets, and rebuild both BCC directions. Retry only invalid expressions once; keep successful siblings. Publish validated keep/rewrite rows with full retained mask sequences. **${rejected} rejected expressions and ${excluded_errors} unresolved validation errors are excluded**, with errors kept distinct in local audits. `train_filtered` adds the final complete-BCC-only selection.

## Before / after mask filtering

| Training version | Expressions | Mask tracklets | Linked text spans |
| --- | ---: | ---: | ---: |
| Before GPT (`train_old`) | ${train_old_rows} | ${train_old_tracklets} | ${train_old_span_links} |
| After GPT (`train_all`) | ${train_all_rows} | ${train_all_tracklets} | ${train_all_span_links} |
| Complete BCC only (`train_filtered`) | ${train_filtered_rows} | ${train_filtered_tracklets} | ${train_filtered_span_links} |

Tracklets count full instance-mask sequences **per expression**, not unique objects across videos. GPT filtering removes **${removed_tracks} tracklets (${mask_reduction}%)** overall, through expression rejection and context-track removal. Within the retained expressions alone, tracklets decrease from **${retained_before_tracks} to ${train_all_tracklets}** (${removed_retained_tracks} removed). No new segmentation is generated during GPT refinement; retained masks remain lossless.

## Columns

`train_old` bundles all 22 original columns below in one table. Both GPT files retain the same information under readable names, with `disposition` unchanged:

| Original columns → GPT names (where changed) | Contents |
| --- | --- |
| `sample_id`, `video_id`, `expression_id`; `dataset → source_dataset`, `split → source_split` | Annotation and original dataset/video identity. |
| `text → caption`, `negative → is_nonexistent`, `disposition` | Final text, intentional-negative flag, BCC status. |
| `cohort → annotation_category`, `annotation_protocol`, `target_source → target_mask_source` | Annotation category/protocol and target provenance. |
| `frame_ids_json`, `frame_files_json → frame_paths_json` | Ordered source-frame IDs and paths. |
| `tracklets_json → object_tracks_json` | Full frame-aligned instance masks, IDs, and track provenance. |
| `groups_json → text_entities_json`, `span_links_json → caption_object_links_json` | Track-to-text groups and text-to-track links. |
| `dataset_root → source_media_root`, `frame_source → source_media_location`, `provenance_warning → provenance_note` | Source-media references and provenance notes. |
| `extraction_json → original_entity_extraction_json`, `sam_prompt_audit_json → segmentation_audit_json`, `runtime_seconds → generation_runtime_seconds` | Original generation/extraction/SAM audit and runtime; not rewritten GPT provenance. |

GPT files also expose native `frame_ids`, `caption_links` (start/end/text/object IDs), `object_count`, `linked_phrase_count`, and `refinement_decision`. JSON-suffixed strings use `json.loads`; offsets are zero-based Unicode positions with exclusive end. `complete_bcc` means representation checks passed, not human semantic verification. Source video frames are not bundled; obtain media under the upstream datasets' terms.

## Batch throughput and future optimization

Observed on **October 11–12, 2026**, using GPT-6 Luna with medium reasoning; counts include validation retries:

| Stage | API requests | Batches | First submission → last API completion | Average requests/min |
| --- | ---: | ---: | --- | ---: |
| Text-only NLP cleanup | 4,596 | 4 | 13 h 43 min | 5.6 |
| Visual refinement | 6,360 | 99 | 5 h 3 min | 21.0 |

These wall-clock averages include queueing, staggered submissions, and retries; the NLP window also includes the earlier conservative scheduler and collection/resume delays. They are observations, not model-speed guarantees. A visual request can cover multiple captions. Individual visual batches took a median **53 minutes**; all 103 batches completed with **zero API request failures or expirations**. Local validation errors remain separate, as described above.

**Future optimization:** the current wrapper already converts frame/overlay JPEGs to **base64 data URLs** and embeds them in each JSON request within the Batch JSONL file. Base64 does not avoid the size limit. Instead, host those same JPEGs and put their **HTTPS image URLs** in the request JSON. Removing embedded image bytes may keep larger batches below the **200 MB input-file limit**, reducing the many size-driven splits. URLs must remain fetchable throughout processing and retries; the 50,000-request and account queue limits still apply. Fewer files do not guarantee faster inference or lower image-token costs. See the [official OpenAI Batch guidance](https://developers.openai.com/api/docs/guides/batch).

## Cost and remaining improvements

The density-stratified 200-video pilot (100 per dataset) estimated **USD ${expected_cost}** for full Batch processing; the conservative estimate including prior work was **USD ${buffered_cost}**, under the **USD 99 aggregate cap**. Recorded completed charges were **USD ${run_cost} for this run**, **USD ${completed_usd} including pilots**; the ledger accounts for **USD ${accounted_usd} including an older USD ${reserved_usd} reservation**. These are recorded usage costs, not a guaranteed future quote.

Two small follow-up improvements remain:

- **“(s)” cleanup:** GPT removed most training occurrences: **${train_old_literal_s_rows} → ${train_all_literal_s_rows} captions** in `train_all` (${train_filtered_literal_s_rows} in `train_filtered`). Evaluation still contains them (2,808 unfiltered / 313 manually reviewed captions). A simple script over HF data can count linked instances and choose singular/plural wording where unambiguous, then rebuild affected span offsets/BCC links. This optional cleanup has not been applied.
- **Backward mask propagation:** use SAM 3.1 to propagate masks backward in time to recover some missing earlier-frame masks, then validate temporal identity. This is a follow-up segmentation step, not neural-network training backpropagation, and is not included here.

**LVOS still needs processing.** Sparse visual sampling can miss errors; the [public 200-video review viewer](https://concor-video-refinement-pilot.daisydv.chatgpt.site) provides examples, not a corpus-wide quality score.
