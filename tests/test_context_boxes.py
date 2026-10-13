import hashlib,json,unittest
from scripts.custom_dataset import pad_mask_context
from scripts.build_baseline_dataset import apply_context_expansions
from pathlib import Path

class ContextBoxTests(unittest.TestCase):
    def fixture(self):
        review=Path('fixture-review').resolve();raw=b'fixture snapshot'
        row={'id':'a','label_status':'verified','human_reviewed':True,'processed_sha256':'pixel-identity',
             'width':100,'height':100,'boxes':[{'class_id':2,'bbox':[20,20,60,60]}]}
        entry={'id':'a','processed_sha256':'pixel-identity','original_boxes':json.loads(json.dumps(row['boxes'])),
               'expanded_boxes':[{'class_id':2,'bbox':[10,15,90,95]}]}
        ledger={'authorization':'user_requested_mask_context','review_sources':{str(review):{
            'manifest_sha256':hashlib.sha256(raw).hexdigest(),'rows':[entry]}}}
        return review,raw,{'images':[row]},ledger
    def test_padding_clamps_and_covers_visible_context(self):
        original=[0,10,90,90];padded=pad_mask_context(original,100,100)
        self.assertEqual(padded,[0,2,100,100])
    def test_export_tracks_base_review_and_separate_assistant_geometry(self):
        review,raw,custom,ledger=self.fixture();original=json.loads(json.dumps(custom))
        apply_context_expansions(review,raw,custom,ledger)
        row=custom['images'][0];self.assertEqual(row['boxes'][0]['bbox'],[10.,15.,90.,95.])
        self.assertEqual(row['context_expansion']['original_boxes'],original['images'][0]['boxes'])
        self.assertFalse(row['context_expansion']['geometry_human_reverified'])
    def test_unverified_changed_classes_shrinking_and_stale_snapshots_rejected(self):
        for mode in ['unverified','class','shrink','stale']:
            review,raw,custom,ledger=self.fixture();entry=ledger['review_sources'][str(review)]['rows'][0]
            if mode=='unverified':custom['images'][0]['human_reviewed']=False
            if mode=='class':entry['expanded_boxes'][0]['class_id']=1
            if mode=='shrink':entry['expanded_boxes'][0]['bbox']=[30,30,50,50]
            if mode=='stale':raw=b'changed snapshot'
            with self.subTest(mode=mode),self.assertRaises(ValueError):apply_context_expansions(review,raw,custom,ledger)

if __name__=='__main__':unittest.main()
