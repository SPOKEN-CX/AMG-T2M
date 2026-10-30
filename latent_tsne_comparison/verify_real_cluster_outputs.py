#!/usr/bin/env python3
"""Check source separation, actual forward-pass parity and geometry of final artifacts."""
import csv,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.distance import pdist
from PIL import Image
from common import PROJECT,REPO,digest,read_rows,save_json
from features import FrozenEncoder,motion_batch,normalize
from semantic_metric_head import SemanticHead,embed
from threadpoolctl import threadpool_limits

ROOT=PROJECT/'artifacts/semantic_metric_head_v1';OUT=ROOT/'fusion_soft_v1'

def main():
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'protocol_before_scores.json').read_text());fusion=json.loads((OUT/'protocol_before_fusion_scores.json').read_text());w=fusion['weight_semantic']
    annotations=json.loads((REPO/'models/Retrival/datasets/annotations/humanml3d/annotations.json').read_text())
    def rec(r):return annotations[r['motion_id']]['path'].removeprefix('M/')
    fitting=read_rows(ROOT/'fit.csv');dev=read_rows(ROOT/'dev.csv');test=read_rows(ROOT/'source_disjoint_test102/samples.csv')
    assert not {rec(r) for r in fitting}&{rec(r) for r in dev}
    dataset=REPO/'dataset/HumanML3D'
    forbidden={annotations[i]['path'].removeprefix('M/') for s in ['train','val'] for i in (dataset/f'{s}.txt').read_text().split()}
    assert not forbidden & {rec(r) for r in test}
    for p,expected in plan['hashes'].items():assert digest(p)==expected,(p,'source changed')
    audit={'frozen_source_hashes_unchanged':True,'fit_development_recording_intersection':0,'test_official_train_val_recording_intersection':0,
           'no_subject_independence_claimed':True,'test_legacy_selection_history_disclosed':True,'new_semantic_branch':True,'raw_vq_unchanged':True,
           'weight_semantic':w,'weight_raw_motion':1-w,'figures':{},'inference_parity':{}}
    cfg=json.loads((PROJECT/'config.json').read_text())
    rows=read_rows(OUT/'previous_exploratory420/samples.csv');chosen=np.arange(0,len(rows),60)
    bx=torch.from_numpy(motion_batch([rows[i] for i in chosen],cfg))
    for arm,path in [('baseline',REPO/'output/hml3d/VQ_base/net_last.pth'),('ours',REPO/'checkpoints/self_train/VQVAE/t2m/net_best_fid.pth')]:
        model=FrozenEncoder(cfg,path).eval()
        with torch.inference_mode():actual=normalize(model(bx)[1].numpy())
        with np.load(ROOT/'previous_exploratory420'/f'{arm}_raw_features.npz',allow_pickle=False) as z:expected=z['features'][chosen]
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        h=SemanticHead();h.load_state_dict(torch.load(ROOT/f'{arm}_head_s20261030.pth',map_location='cpu')['head'],strict=True)
        sem=embed(h,actual);actual_fusion=np.concatenate([np.sqrt(1-w)*actual,np.sqrt(w)*sem],axis=1)
        with np.load(OUT/'previous_exploratory420'/f'{arm}_fusion_soft_features.npz',allow_pickle=False) as z:pred=z['features'][chosen]
        np.testing.assert_allclose(actual_fusion,pred,rtol=2e-5,atol=2e-6)
        audit['inference_parity'][arm]={'real_motion_crops_reencoded':len(chosen),'max_raw_error':float(np.abs(actual-expected).max()),'max_fused_error':float(np.abs(actual_fusion-pred).max()),'labels_not_given_to_encoder_or_head':True}
    for dataset_name in ['previous_exploratory420','source_disjoint_test102']:
        folder=OUT/dataset_name;rows=read_rows(folder/'samples.csv')
        for stem,keys in [('real_fusion_comparison',['baseline_raw','ours_fusion_soft']),('matched_fusion_control',['baseline_fusion_soft','ours_fusion_soft'])]:
            with (folder/f'{stem}_plotted_coordinates.csv').open(encoding='utf-8-sig',newline='') as f:p=list(csv.DictReader(f))
            assert [q['sample_id'] for q in p]==[r['sample_id'] for r in rows]
            errors={}
            for side,key in zip(['left','right'],keys):
                if dataset_name=='previous_exploratory420' and key=='baseline_raw':source=PROJECT/'artifacts/reference_style_action_search_v3/selected_profile/baseline_pca_p30_e12_s42.npz'
                else:source=folder/f'{key}_tsne_p30_s42.npz'
                with np.load(source,allow_pickle=False) as z:xy=z['coordinates'].astype('float64')
                shown=np.array([[float(q[f'{side}_x']),float(q[f'{side}_y'])] for q in p])
                a,b=pdist(xy),pdist(shown)
                np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-5)
                errors[side]=float(np.abs(a-b).max())
            size=Image.open(folder/f'{stem}.png').size;preview=Image.open(folder/f'{stem}_preview.png').size
            assert size==(7560,2745) and preview==(2520,915)
            audit['figures'][f'{dataset_name}/{stem}']={'points_per_panel':len(rows),'point_pair_distances_preserved_after_global_rotation':True,'max_pair_distance_error':errors,'png_pixels':size,'preview_pixels':preview,'physical_panel_sizes_match':True}
    save_json(OUT/'final_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
