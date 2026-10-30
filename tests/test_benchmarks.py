import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from aod.data.amber import load_amber_discriminative
from tools.runtime import group_split,binary_metrics

class BenchmarkTests(unittest.TestCase):
    def test_official_amber_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,a=Path(tmp)/'q.json',Path(tmp)/'a.json'
            types=['generative','discriminative-hallucination','discriminative-attribute-state','discriminative-relation','relation']
            q.write_text(json.dumps([dict(id=i,image=f'{i}.jpg',query='Question?') for i in range(5)]))
            a.write_text(json.dumps([dict(id=i,type=t,truth=['cat'] if i==0 else 'yes') for i,t in enumerate(types)]))
            rows=load_amber_discriminative(str(q),str(a))
            self.assertEqual([r.typology for r in rows],['existence','attribute','relation','relation'])

    def test_images_stay_in_one_split(self):
        rows=[dict(id=i,image=f'{i//3}.jpg') for i in range(300)]
        chosen=group_split(rows,50,42)
        train={r['image'] for r in chosen if r['split']=='train'}
        test={r['image'] for r in chosen if r['split']=='test'}
        self.assertEqual((len(train),len(test)),(40,10));self.assertFalse(train&test)
        self.assertEqual(chosen,group_split(rows,50,42))

    def test_confusion_counts(self):
        rows=[dict(id=i,gt=g,pred=p) for i,(g,p) in enumerate([(1,1),(0,1),(0,0),(1,0)])]
        m=binary_metrics(rows);self.assertEqual(m['f1'],50);self.assertEqual(m['false_positive_rate'],50)
        with self.assertRaises(ValueError):binary_metrics(rows+rows)
        with self.assertRaises(ValueError):binary_metrics([])

    def test_amber_no_is_positive(self):
        from amber.evaluate import evaluate_records
        rows=[dict(id=i,gt=g,pred=p,base_pred=p,typology='existence')
              for i,(g,p) in enumerate([(0,0),(0,0),(0,1),(1,1)])]
        m=evaluate_records(rows)
        self.assertEqual(m['base']['positive_label'],'no')
        self.assertEqual(m['base']['f1'],80)
        self.assertAlmostEqual(m['yes_positive_auxiliary']['base']['f1'],200/3)

    def test_chair_compounds_and_occurrences(self):
        from chair.official_algorithm import CHAIR
        from chair.evaluate import score
        e=CHAIR([1],'');e.imid_to_objects[1]={'person','hot dog'}
        words,nodes,_,_=e.caption_to_words('A person eats two hot dogs. A dog and another dog watch.')
        self.assertEqual(nodes.count('dog'),2);self.assertIn('hot dog',nodes)
        m=score(e,[dict(image_id=1,caption='A dog and another dog watch a person.')],'caption')
        self.assertEqual(m['object_mentions'],3);self.assertEqual(m['hallucinated_mentions'],2)
        self.assertAlmostEqual(m['CHAIRi'],200/3);self.assertEqual(m['CHAIRs'],100)


class AnswerTokenTests(unittest.TestCase):
    def test_word_start_answers_and_fail_closed(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('answers',ROOT/'amber/src/aod/vlm/answers.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):
                return {'Yes':[8241],'No':[3782],' Yes':[3869],' No':[1939]}[text]
        self.assertEqual(module.llava_answer_ids(Tokenizer()),(3869,1939))
        class Incompatible:
            def encode(self,text,add_special_tokens=False):return [1,2]
        with self.assertRaises(ValueError):module.llava_answer_ids(Incompatible())

if __name__=='__main__':unittest.main()
