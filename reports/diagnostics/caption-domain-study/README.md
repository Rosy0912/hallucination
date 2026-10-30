# Additional training/validation diagnostic

This study collected causal hidden features before object mentions in Base captions on 60 training images and the same 16 validation images. COCO object consistency supplied the labels. It trained a scalar projection probe with a residual adversary.

The run completed on one A100. All 16 validation Base captions exactly matched the main CHAIR run. There were 507 training mentions (72 inconsistent) and 121 validation mentions (17 inconsistent), with no skipped alignment cases. Best validation object-consistency AUROC was 0.8857. This target differs from the question-consistency probe, so their AUROCs are not directly comparable.

No test images were accessed, no caption-decoding benchmark result is claimed for this checkpoint, and this study did not select or replace the checkpoint used for the reported 50-image CHAIR result. The main package retains its predeclared validation selection. This archive preserves the additional diagnostic rather than presenting it as a successful benchmark experiment.

collection.py is the actual collection/training source. It requires a prepared standalone CHAIR package plus its documented model, image, annotation and NLTK resources; its --help lists the explicit paths. It does not execute final-test evaluation.
