"""Image-conditioned object checks with the POPE-trained raw L13 readout.

This is a composite captioning method, not the original AOD decoder alone.
No image annotations are loaded by this module at inference time.
"""
from pathlib import Path
import torch
from torch import nn
from aod.vlm.intervention import resolve_decoder_layers, forward_next_logits, greedy_generate_ids
from tools.runtime import inputs_for

REWRITE = ('Describe this image in a detailed paragraph. An image-based check supports '
           'these object types: {objects}. Focus on the listed objects that are visible, '
           'describing their appearance, actions, placement and spatial relationships '
           'in as much visible detail as possible. Avoid introducing additional object '
           'types or guessing. Output only the description.')

class ObjectVerifier:
    def __init__(self, checkpoint, model, processor):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['variant'] != 'projective' or payload['layer'] != 13 or payload['mu'] is not None:
            raise ValueError('Expected the POPE projective raw L13 checkpoint')
        torch.set_float32_matmul_precision('high')
        self.head = nn.Sequential(nn.Linear(4096, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1))
        weights = {key.removeprefix('truth_probe.'): value for key, value in payload['state_dict'].items()
                   if key.startswith('truth_probe.')}
        self.head.load_state_dict(weights)
        self.head = self.head.cuda().eval()
        self.model, self.processor = model, processor
        layers, _ = resolve_decoder_layers(model)
        self.layer = layers[12]
        from official_algorithm import CHAIR
        self.lexicon = CHAIR([], '/not-used-for-inference')

    @torch.inference_mode()
    def checks(self, image, caption):
        objects = sorted(set(self.lexicon.caption_to_words(caption)[1]))
        checks = []
        for obj in objects:
            article = 'an' if obj[0] in 'aeiou' else 'a'
            question = f'Is there {article} {obj} in the image?'
            captured = {}
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured['hidden'] = hidden[:, -1, :].detach().float().clone()
            handle = self.layer.register_forward_hook(hook)
            try:
                forward_next_logits(self.model, inputs_for(self.processor, image, question))
            finally:
                handle.remove()
            checks.append(dict(object=obj, question=question,
                               logit=float(self.head(captured['hidden']).item())))
        return checks

    @torch.inference_mode()
    def rewrite(self, image, row, checks, threshold, decoding):
        keep = [item['object'] for item in checks if item['logit'] >= threshold]
        reject = [item['object'] for item in checks if item['logit'] < threshold]
        result = dict(checks=checks, rejected=reject, rewritten=bool(keep and reject))
        if result['rewritten']:
            prompt = REWRITE.format(objects=', '.join(keep))
            result['rewrite_prompt'] = prompt
            inputs = inputs_for(self.processor, image, prompt)
            for key, config in [('verified_base', None), ('aod', decoding)]:
                ids = greedy_generate_ids(self.model, inputs, cfg=config, max_new_tokens=128,
                                          eos_token_ids=[self.processor.tokenizer.eos_token_id])
                result[key+'_caption'] = self.processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                result[key+'_tokens'] = ids.shape[1]
        else:
            for key, source in [('verified_base', 'base'), ('aod', 'anchor')]:
                for suffix in ['caption', 'tokens']:
                    result[key+'_'+suffix] = row[source+'_'+suffix]
        return result
