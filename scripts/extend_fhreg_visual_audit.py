"""Read-only patch evidence and identity null diagnostic, without any extra network call."""
import json
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from sglm.evaluation import evaluate_transform
from sglm.experiment_io import write_csv


def main():
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);private=Path(p['private']);manifest=json.loads((out/'evidence_manifest.json').read_text())
    sheets={k:[] for k in p['sample_selection']['categories']};rows=[]
    for record in manifest['pairs_manifest']:
        folder=Path(record['folder']);c=np.load(folder/'full_correspondence_evidence.npz');opt=np.array(Image.open(folder/'optical_original.png').convert('L'));sar=np.array(Image.open(folder/'sar_augmented_actual_input.png').convert('L'))
        ii=np.flatnonzero(c['RANSAC_inlier_mask']);chosen=ii[np.unique(np.linspace(0,len(ii)-1,min(6,len(ii))).astype(int))] if len(ii) else []
        panel=Image.new('RGB',(6*170,125),'white');draw=ImageDraw.Draw(panel)
        draw.text((4,3),record['pair_id']+' | optical(left)/SAR_aug(right), fixed uniform six inliers, no GT selection',fill='black')
        for j,k in enumerate(chosen):
            patches=[]
            for im,xy in [(opt,c['filtered_source'][k]),(sar,c['filtered_target'][k])]:
                patch=cv2.getRectSubPix(im,(41,41),tuple(map(float,xy)));patch=cv2.resize(patch,(80,80),interpolation=cv2.INTER_NEAREST);patches.append(patch)
            pair=np.concatenate(patches,axis=1);panel.paste(Image.fromarray(pair).convert('RGB'),(j*170,28))
            # Crosshair only for actual predicted locations; no inferred/handplaced landmarks.
            for x in [j*170+40,j*170+120]:draw.line((x-4,68,x+4,68),fill='red',width=1);draw.line((x,64,x,72),fill='red',width=1)
            draw.text((j*170+3,110),f'filtered id {k}',fill='black')
        if not len(ii):draw.text((20,65),'NO RANSAC INLIERS; no substitute correspondences',fill='red')
        panel.save(folder/'correspondence_patch_review.png');sheets[record['category']].append(panel)
        M=c['matrix'];probe=c['probes'];gt=c['nominal_probe_truth'];identity=evaluate_transform(np.array([[1.,0,0],[0,1,0]]),probe,gt)
        rows.append(dict(pair_id=record['pair_id'],FHReg_probe_rmse_px=record['evaluation']['rmse_px'],identity_probe_rmse_px=identity['rmse_px'],
            identity_SR5=identity['success']['5'],FHReg_SR5=record['evaluation']['success']['5'],FHReg_has_transform=M.shape==(2,3),
            FHReg_vs_identity_probe_max_displacement_px=float(np.linalg.norm(probe@M[:,:2].T+M[:,2]-probe,axis=1).max()) if M.shape==(2,3) else None,
            uniformly_displayed_filtered_inlier_ids=list(map(int,chosen))))
    for category,panels in sheets.items():
        sheet=Image.new('RGB',(1020,125*len(panels)),'white')
        for i,panel in enumerate(panels):sheet.paste(panel,(0,125*i))
        sheet.save(private/f'{category}_patch_sheet.png')
    write_csv(out/'random30_identity_diagnostic.csv',[{**r,'uniformly_displayed_filtered_inlier_ids':json.dumps(r['uniformly_displayed_filtered_inlier_ids'])} for r in rows])
    (out/'patch_review_manifest.json').write_text(json.dumps(dict(pairs=30,uniform_inlier_patches_per_pair=6,no_GT_selection=True,no_network=True,
        note='Contrast/texture patches are qualitative, not independent physical GT. Patch display is resized and must not be used for geometric measurement.',pairs_manifest=rows),indent=2))
    # Same existing independent probes, available for every originalT05validation pair.
    source=json.loads(Path(p['source_protocol']).read_text());fh=json.loads((Path(source['source_root'])/'FHReg_GUSO_validation.json').read_text());byid={r['pair_id']:r for r in fh['pairs']}
    cache=json.loads(Path('reports/window_oracle_v1/cache_freeze.json').read_text());identity_rows=[]
    for row in cache['pairs']:
        c=np.load(row['path']);r=byid[row['pair_id']];ev=evaluate_transform(np.array([[1.,0,0],[0,1,0]]),c['probes'],c['probe_truth']);M=r['estimation']['matrix']
        identity_rows.append(dict(pair_id=row['pair_id'],identity_probe_RMSE_px=ev['rmse_px'],identity_SR3=ev['success']['3'],identity_SR5=ev['success']['5'],FHReg_probe_RMSE_px=r['evaluation']['rmse_px'],FHReg_SR5=r['evaluation']['success']['5'],
            FHReg_matrix_identity_within_1e_minus6=bool(M is not None and np.max(np.abs(np.array(M)-np.array([[1.,0,0],[0,1,0]])))<1e-6)))
    write_csv(out/'identity_null_all952.csv',identity_rows)
    (out/'identity_null_summary.json').write_text(json.dumps(dict(pairs=952,no_network=True,no_refit=True,does_not_change_published_table=True,
        identity_SR3=sum(r['identity_SR3'] for r in identity_rows)/952,identity_SR5=sum(r['identity_SR5'] for r in identity_rows)/952,FHReg_SR5=sum(r['FHReg_SR5'] for r in identity_rows)/952,
        FHReg_near_identity_matrices=sum(r['FHReg_matrix_identity_within_1e_minus6'] for r in identity_rows),FHReg_success_identity_failed=sum(r['FHReg_SR5'] and not r['identity_SR5'] for r in identity_rows),
        interpretation='Rules out fixedidentity transform as explanation of aggregateFH score; does not prove every match is physical or modeldependscausally on pixels.'),indent=2))
    print('Patch review30 and identity-null952 complete',flush=True)


if __name__=='__main__':main()
