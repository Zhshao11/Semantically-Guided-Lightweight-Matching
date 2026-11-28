"""Thirty frozen random pairs: raw/filtered/inlier/probe/warp evidence; no threshold tuning."""
import argparse,hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sglm.sar_references import SARReference
from sglm.sar_reference_geometry import restore_resized_coordinates,filter_correspondences
from sglm.external_reference import cpu_input
from sglm.t02.data import AffinePairs256
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform
from sglm.geometry import transform_points
from sglm.diagnostics import bn_snapshot,bn_unchanged,coverage
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gray_rgb(image):
    gray=image[:,:,0] if image.ndim==3 else image
    return np.repeat(gray[:,:,None],3,2)


def match_plot(opt,sar,source,target,path,title,score=None,max_lines=80):
    source=np.asarray(source).reshape(-1,2);target=np.asarray(target).reshape(-1,2);fig,axs=plt.subplots(1,2,figsize=(9,4.8))
    for ax,img in zip(axs,[opt,sar]):ax.imshow(img,cmap='gray');ax.set_xlim(-.5,img.shape[1]-.5);ax.set_ylim(img.shape[0]-.5,-.5);ax.set_aspect('equal');ax.tick_params(labelsize=7)
    # Save all coordinates, display all point locations plus uniformly sampled lines.
    axs[0].scatter(source[:,0],source[:,1],s=1,color='yellow',alpha=.35);axs[1].scatter(target[:,0],target[:,1],s=1,color='yellow',alpha=.35)
    selected=np.unique(np.linspace(0,len(source)-1,min(max_lines,len(source))).astype(int)) if len(source) else []
    for k in selected:
        color=plt.cm.hsv((int(k)*.61803398875)%1);fig.add_artist(ConnectionPatch(source[k],target[k],coordsA='data',coordsB='data',axesA=axs[0],axesB=axs[1],color=color,alpha=.65,linewidth=.6))
    axs[0].set_title('Optical');axs[1].set_title('SAR augmented model input')
    fig.suptitle(title,fontsize=10);fig.text(.02,.015,f'All{len(source)} point locations;{len(selected)} uniformly selected lines for display only. No GT-based selection.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.94));fig.savefig(path,dpi=130);plt.close(fig)


def overlay_images(optical,sar_aug,matrix,sar_mask):
    optical=gray_rgb(optical);sar_aug=gray_rgb(sar_aug)
    if matrix is None:
        placeholder=Image.fromarray(optical);draw=ImageDraw.Draw(placeholder);draw.rectangle((0,0,255,25),fill='black');draw.text((5,8),'FIT FAILED: NO TRANSFORM',fill='red')
        return np.array(placeholder),np.array(placeholder),None,None
    inverse=cv2.invertAffineTransform(np.asarray(matrix,np.float64));warped=cv2.warpAffine(sar_aug,inverse,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.warpAffine(sar_mask.astype(np.float32),inverse,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.999
    overlay=np.stack([optical[:,:,0],warped[:,:,0],warped[:,:,0]],-1);overlay[~valid]=optical[~valid]//3
    yy,xx=np.indices((256,256));choice=((xx//32+yy//32)%2).astype(bool)&valid;checker=optical.copy();checker[choice]=warped[choice]
    return overlay,checker,warped,valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);private=Path(p['private']);freeze=json.loads((out/'audit_freeze.json').read_text())
    for file,digest in freeze['inputs_sha256'].items():assert sha(file)==digest,file
    source=json.loads(Path(p['source_protocol']).read_text());cfg=json.loads(Path(source['frozen']+'/candidate_config.json').read_text());selection=json.loads((out/'random30_selection.json').read_text())
    original={r['pair_id']:r for r in json.loads((Path(source['source_root'])/'FHReg_GUSO_validation.json').read_text())['pairs']}
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=SARReference('FHReg_GUSO',source,cfg);before=bn_snapshot(model.model);ds=AffinePairs256(a.root,source['frozen']+'/validation_plan.json',split='val',preload=False);index_by_id={r['pair_id']:i for i,r in enumerate(ds.samples)}
    all_records=[];cards={category:[] for category in p['sample_selection']['categories']}
    with torch.inference_mode():
        for selected in selection['pairs']:
            uid=selected['pair_id'];index=index_by_id[uid];item=ds[index];data=cpu_input(item);opt_original,sar_original=ds.read_images(ds.samples[index]);folder=private/'evidence'/selected['category']/f"{selected['draw_order']+1:02d}_pair{uid.split('/')[-1]}";folder.mkdir(parents=True,exist_ok=True)
            inputs=model.prepare(data);batch=model.model(dict(image_opt=inputs[0],image_sar=inputs[1]))
            raw_native=[batch[key].float().cpu().numpy() for key in ['mkpts0_f','mkpts1_f','mconf_f']]
            raw_source=restore_resized_coordinates(raw_native[0],(256,256),(512,512));raw_target=restore_resized_coordinates(raw_native[1],(256,256),(512,512));raw_score=raw_native[2]
            src,dst,score,valid=filter_correspondences(raw_source,raw_target,raw_score,data['optical_mask'],data['sar_mask'])
            repeat=model.predict(data);assert all(np.array_equal(value,repeat[key]) for value,key in [(src,'source'),(dst,'target'),(score,'score')])
            fit=estimate_from_correspondences(src,dst,score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            ref=original[uid];assert len(raw_score)==ref['raw_matches'] and len(score)==ref['valid_matches'] and fit['matrix']==ref['estimation']['matrix'] and fit['inliers']==ref['estimation']['inliers']
            # GT accessed only after model prediction, common filter and original fit.
            probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
            ev=evaluate_transform(fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=fit['status']);assert ev==ref['evaluation']
            ii=np.array(fit['inlier_indices'],int);inlier_mask=np.zeros(len(src),bool);inlier_mask[ii]=True
            predicted_probes=transform_points(probes,fit['matrix']) if fit['matrix'] is not None else np.full_like(probes,np.nan)
            nominal_dst=transform_points(src,item['A_forward'].numpy());corr_errors=np.linalg.norm(dst-nominal_dst,axis=1)
            Image.fromarray(opt_original).save(folder/'optical_original.png');Image.fromarray(sar_original).save(folder/'sar_original_unaugmented.png');Image.fromarray(data['sar']).save(folder/'sar_augmented_actual_input.png')
            for name,im in [('FHReg_optical_native512.png',cv2.resize(opt_original,(512,512))),('FHReg_SAR_aug_native512.png',cv2.resize(data['sar'],(512,512)))]:Image.fromarray(im).save(folder/name)
            match_plot(opt_original,data['sar'],raw_source,raw_target,folder/'matches_raw.png',f'{uid}: native model outputs, before common filter')
            match_plot(opt_original,data['sar'],src,dst,folder/'matches_filtered.png',f'{uid}: common filtered predictions')
            match_plot(opt_original,data['sar'],src[ii],dst[ii],folder/'ransac_inliers.png',f'{uid}: original affine RANSAC inliers')
            fig,axs=plt.subplots(1,2,figsize=(9,4.8));axs[0].imshow(opt_original);axs[1].imshow(data['sar'],cmap='gray')
            axs[0].scatter(probes[:,0],probes[:,1],s=8,color='cyan',marker='+');axs[1].scatter(truth[:,0],truth[:,1],s=8,color='cyan',marker='+',label='nominal GT probe')
            if fit['matrix'] is not None:axs[1].scatter(predicted_probes[:,0],predicted_probes[:,1],s=6,color='magenta',marker='x',label='predicted affine probe')
            for ax in axs:ax.set_xlim(-.5,255.5);ax.set_ylim(255.5,-.5);ax.set_aspect('equal')
            axs[0].set_title('Fixed independent optical probes');axs[1].set_title('SAR_aug: nominal + / estimated x');axs[1].legend(fontsize=7);fig.suptitle(f'{uid}: probes never fitted');fig.tight_layout();fig.savefig(folder/'independent_probes.png',dpi=130);plt.close(fig)
            overlay,checker,warped,warp_valid=overlay_images(opt_original,data['sar'],fit['matrix'],data['sar_mask'])
            Image.fromarray(overlay).save(folder/'registered_overlay.png');Image.fromarray(checker).save(folder/'registered_checkerboard.png')
            if warped is not None:Image.fromarray(warped).save(folder/'SAR_registered_to_optical.png');Image.fromarray(warp_valid.astype(np.uint8)*255).save(folder/'registration_valid_mask.png')
            gt_overlay,gt_checker,_,_=overlay_images(opt_original,data['sar'],item['A_forward'].numpy(),data['sar_mask']);Image.fromarray(gt_overlay).save(folder/'nominal_GT_overlay_diagnostic_only.png')
            np.savez_compressed(folder/'full_correspondence_evidence.npz',native_source512=raw_native[0],native_target512=raw_native[1],raw_source256=raw_source,raw_target256=raw_target,raw_score=raw_score,
                common_filter_mask=valid,filtered_source=src,filtered_target=dst,filtered_score=score,RANSAC_inlier_mask=inlier_mask,probes=probes,nominal_probe_truth=truth,predicted_probes=predicted_probes,
                matrix=np.array(fit['matrix']) if fit['matrix'] is not None else np.empty((0,)),A_forward=item['A_forward'].numpy())
            write_csv(folder/'raw_matches_all.csv',[dict(raw_id=k,opt_x512=float(raw_native[0][k,0]),opt_y512=float(raw_native[0][k,1]),sar_aug_x512=float(raw_native[1][k,0]),sar_aug_y512=float(raw_native[1][k,1]),
                opt_x256=float(raw_source[k,0]),opt_y256=float(raw_source[k,1]),sar_aug_x256=float(raw_target[k,0]),sar_aug_y256=float(raw_target[k,1]),confidence=float(raw_score[k]),common_filter_keep=bool(valid[k])) for k in range(len(raw_score))])
            write_csv(folder/'filtered_matches_all.csv',[dict(filtered_id=k,opt_x=float(src[k,0]),opt_y=float(src[k,1]),sar_aug_x=float(dst[k,0]),sar_aug_y=float(dst[k,1]),confidence=float(score[k]),ransac_inlier=bool(inlier_mask[k]),nominal_correspondence_error_px=float(corr_errors[k])) for k in range(len(score))])
            write_csv(folder/'independent_probes_all.csv',[dict(probe_id=k,opt_x=float(probes[k,0]),opt_y=float(probes[k,1]),nominal_sar_x=float(truth[k,0]),nominal_sar_y=float(truth[k,1]),predicted_sar_x=float(predicted_probes[k,0]) if fit['matrix'] is not None else None,predicted_sar_y=float(predicted_probes[k,1]) if fit['matrix'] is not None else None) for k in range(len(probes))])
            record=dict(**selected,index=index,folder=str(folder),prediction_and_fit_exact_original=True,source_sha256=ds.samples[index]['optical_sha256'],sar_original_sha256=ds.samples[index]['sar_sha256'],
                raw_matches=len(raw_score),filtered_matches=len(score),removed=int((~valid).sum()),inliers=fit['inliers'],fit=fit,evaluation=ev,
                inlier_source_coverage=coverage(src[ii]),nominal_filtered_correct_fraction3=float((corr_errors<3).mean()) if len(score) else None,
                nominal_filtered_correct_fraction5=float((corr_errors<5).mean()) if len(score) else None,nominal_inlier_correct_fraction5=float((corr_errors[ii]<5).mean()) if len(ii) else None,
                overlay_direction='warp augmentedSAR -> optical using inverse(predicted opt->SAR_aug affine); no GTfallback for failed fits',
                physical_correspondence_verification='qualitative visual review pending; nominal GT errors not independent physical labels',
                files_sha256={file.name:sha(file) for file in folder.iterdir() if file.is_file()})
            (folder/'metadata.json').write_text(json.dumps(record,indent=2));all_records.append(record)
            # Compact review card: all30 are later viewed; no result-dependent case replacement.
            card=Image.new('RGB',(256*4,292),'white');draw=ImageDraw.Draw(card)
            for k,im in enumerate([gray_rgb(opt_original),gray_rgb(data['sar']),overlay,checker]):card.paste(Image.fromarray(im),(256*k,30))
            error='FAIL' if ev['rmse_px'] is None else f"{ev['rmse_px']:.3f}"
            draw.text((5,2),f"{selected['draw_order']+1:02d} {uid} | FH RMSE={error}, T05={selected['T05_rmse_px']:.3f} | raw/keep/inlier={len(raw_score)}/{len(score)}/{fit['inliers']}",fill='black')
            for k,label in enumerate(['Optical original','SAR_aug actual input','Estimated warp overlay','Estimated checkerboard']):draw.text((256*k+5,17),label,fill='black')
            card.save(folder/'review_card.png');cards[selected['category']].append(card)
            print(selected['category'],selected['draw_order']+1,uid,'exact',flush=True)
    assert bn_unchanged(before,model.model)
    for category,images in cards.items():
        sheet=Image.new('RGB',(1024,292*len(images)),'white')
        for k,im in enumerate(images):sheet.paste(im,(0,k*292))
        sheet.save(private/f'{category}_contact_sheet.png')
    (out/'evidence_manifest.json').write_text(json.dumps(dict(passed=True,pairs=30,per_category=10,random_selection_sha256=sha(out/'random30_selection.json'),all30_predictions_fits_evaluations_exact_previous=True,
        all_modules_eval=True,BN_unchanged=True,raw_points_are_native_post_network_pre_common_filter=True,probe_GT_used_only_after_fit=True,
        raw_original_SAR_and_actual_SAR_aug_saved_separately=True,failed_fits_never_use_GT_overlay_as_prediction=True,all_coordinates_saved_not_only_display_subsample=True,
        private_contact_sheets=[str(private/f'{category}_contact_sheet.png') for category in cards],pairs_manifest=all_records),indent=2))


if __name__=='__main__':main()
