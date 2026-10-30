# Sources and versions

AOD code: user-supplied MIT-licensed AOD-main / AOD-evaluation-20260929, upstream https://github.com/Hunter-Wrynn/AOD .

POPE: https://github.com/RUCAIBox/POPE . Fixed user-provided question/group manifests are included where needed.
HallusionBench: https://github.com/tianyi-lab/HallusionBench . Images: https://huggingface.co/datasets/lmms-lab-encoder/HallusionBench , revision cd417161857aefb23d878d42cf1bb53aa9dd646f; downloaded bytes and question/label equality are checked.
AMBER: https://github.com/junyangwang0410/AMBER (Apache-2.0); official query/annotation files and Google Drive image archive.
COCO: https://cocodataset.org/ ; public val2014 images and train/val2014 instances/captions annotations.
LLaVA: https://huggingface.co/llava-hf/llava-1.5-7b-hf , revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9. See the model's license and model card.

CHAIR scoring adaptation (CHAIR package): https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py and original synonyms. Python 3 adaptation preserves token/compound/ground-truth/mention-count rules. Pattern singularization code: https://github.com/clips/pattern/blob/master/pattern/text/en/inflect.py ; BSD license included as PATTERN_LICENSE.txt.

No images, base-model weights, credentials or private feature caches are distributed in the standalone source archive. The included LICENSE applies to the AOD/wrapper code; third-party data/model/source terms remain applicable.
