# Tracklet-native correspondence pipeline

The linguistic unit is an exact half-open character span `[start, end)`. The visual unit is a tracklet: one instance ID with a frame-aligned sequence of COCO RLE masks. The temporal dimension changes the visual representation, not the correspondence rules.

## Main referent

For Ref-YouTube-VOS train and nonnegative ReVOS records, the official target masks remain authoritative. They are never rediscovered or replaced. If a Ref-YouTube-VOS split has no public masks, SAM3.1 is prompted for the main referring span; those tracks use source `sam3.1_main_referent` and are not presented as ground truth.

One text span may identify multiple official or predicted tracks. This is needed for plural referring expressions. A single tracklet may also be linked from the main noun phrase and a later pronoun or possessive.

## Context extraction

spaCy parses each original expression. The extractor keeps articles and modifiers in the linked surface span while generating a short normalized SAM prompt. Independently visible people, objects, garments, body parts, surfaces, and stuff regions are eligible context.

Scene/meta words, events, taxonomic ranks, quoted titles, comparison classes, viewpoint `camera`, and generic `thing(s)`/`object` are not sent as semantic prompts. Every ignored candidate and reason remains in `extraction_json`; the decision is inspectable rather than silent.

## SAM3.1 and filtering

A long-lived SAM3.1 Object Multiplex predictor is loaded once per worker. Frames are materialized once in stable numeric order. Each unique prompt is issued once and all matching instances returned for that prompt are tracked together.

Tracks are retained when they:

- appear in at least one frame for videos of four frames or fewer, otherwise at least `max(2, ceil(5% × frames))` frames;
- contain at least 64 foreground pixels over time;
- have maximum confidence at least 0.55 and mean confidence at least 0.45.

Context tracks overlapping a main referent at temporal volume IoU ≥ 0.65 are removed. Context tracks overlapping an already retained context at IoU ≥ 0.80 are merged, and the merged track retains every prompt and text span.

## Dispositions

- `complete_bcc`: main referent exists and every required context prompt returned at least one retained track.
- `incomplete_context`: main referent exists, but at least one required context prompt returned no retained track.
- `missing_main_referent`: a split without public ground truth produced no retained SAM main track.
- `negative_unsegmentable`: an intentional ReVOS nonexistent-object negative with no tracklets.

These labels describe pipeline completeness, not human-verified semantic correctness.

## Recovery and scaling

Each expression is an atomic JSON checkpoint. Array workers use stable modulo sharding and exclusive claim files, so concurrent workers do not duplicate samples. Generated frames are reused. A preemption signal stops new claims, finishes and fsyncs the active sample, then requeues the same worker. The final exporter can run after any array outcome and records incomplete work as `pending` or `failed` rather than hiding it.
