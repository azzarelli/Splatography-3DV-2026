import os
import json
import matplotlib.pyplot as plt

root = '/home/barry/Desktop/mywork/GS_Research/output/dynerf/'
folder = 'flame_steak/'

focal_word = 'test3' 
bench_marks = ['test','bez_post16k','bez_knnsq','bez_highthres','bez','h_ns_coarse','h_pgns_coarse','h_pgloss','h_abs','w_relu','w_minsq','w_sloth','w_minimize']
#
# Corrections needed here
# '4dgs_noscales','4dgs_opac_hreg_dloss','4dgs_noscales','4dgs_opac_hreg_pg_ngs','4dgs_opac_hreg_pg','4dgs_opac_w_reg','4dgs_corrected','4dgs_densify','prob_6k_w9_neug_nodssim','prob_neug_dynweight','prob_neug_wowx_densification','prob_neug_pgtit','prob_neug_linDens']

# Sparse View
# bench_marks = ['prob_neug_dynweight_25timeres','prob_neug_dynweight','prob_neug_wowx_densification','prob_neug_pgtit','prob_neug_linDens','prob_neug_rigid','prob_neug_pg5','sparsetest','prob_6k_w9_pg','prob_6k_w9_neug_nodssim', 'prob_3k_w9_pg_nodssim','prob_6k_w9_hemb_nos', 'prob_6k_w9_coarse','prob_6k_w9_gradthresh','prob_6k','prob_6k_w9_hemb', 'normal','test_full', 'test', 'Sparse_IsolatedMotionTriPlane', "Sparse_IsolatedMotionTriPlane_01"] # ['l10005TuningNoActivPruneHW','reg_h1','reg_h1_median', 'reg_h01', 'cleanrun', 'PruneHW_l10005_redo', 'W3PruneHW_l10005_WaveLvl']
   
# 'Hthresh00001_NoActivPruneW', 'Hthresh001_NoActivPruneW', 'Hthresh01_NoActivPruneW', 'Hthresh1_NoActivPruneW',]
# #'Wthresh1_NoActivPruneW', 'Wthresh01_NoActivPruneW','Wthresh001_NoActivPruneW','Wthresh0001_NoActivPruneW'] 
# # 'HSUM001_NoActivPruneHW', 'HSUM0_NoActivPruneHW', 'HSUM1_NoActivPruneHW', 'HSUM01_NoActivPruneHW']
"""
benchmark description
l10005TuningNoActivPruneHW:
    Tested l1 weight with 0.00005
    Pruning using the HW method with threshold 0.005
    No opacity embedding or activation
"""
non_words = ['tv', 'HW', 'Shared','NoCos','4-D','OpacEm', 'Bez','Seq',  'Fix','ts','sep', 'tv' 'Feat', 
             'temb', 'Test', 'test','reset', 'Sep', 'Activation', 'd0', 'rotation', 'LR', 'Reg', '12k']

root =  os.path.join(root, folder)

runs = os.listdir(root)

x = {}
y_psnr = []
y_ssim = []
names = []

for run in runs:
    directory = os.path.join(root, run)
    directory = os.path.join(directory, 'active_results')
    check = 0
    for nw in non_words:
        if nw in run:
            check = 1
            break
    
    if run in bench_marks or run == focal_word: # check == 0 or 
        final_check = 0
        d_psnr = []
        d_ssim = []

        names.append(run)
        to_sort = []

        for file in os.listdir(directory):
            file_ = os.path.join(directory, file)

            to_sort.append(int(file.split('.')[0]))

            try:
                with open(file_) as f:
                    d = json.load(f)

                d_psnr.append(d['psnr'])
                d_ssim.append(d['ssim'])
            except:
                d_psnr.append(0.)
                d_ssim.append(0.)

        paired_list = list(zip(to_sort, d_psnr, d_ssim))
        paired_list.sort(key=lambda x: x[0])
        y_psnr.append([x[1] for x in paired_list])
        y_ssim.append([x[2] for x in paired_list])
        

fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

for p,s, name in zip(y_psnr, y_ssim, names):
    if name == focal_word:
        axes[0].plot(p, label=f"{name}", color='black')
        axes[1].plot(s, label=f"{name}", color='black')
    else:
        axes[0].plot(p, label=f"{name}")
        axes[1].plot(s, label=f"{name}") 

axes[0].set_title("PSNR")
axes[0].legend()  # Show the legend
axes[0].grid(True)  # Add a grid

axes[1].set_title("SSIM Results")
axes[1].legend()  # Show the legend
axes[1].grid(True)  # Add a grid

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

    