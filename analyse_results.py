import os
import json
import matplotlib.pyplot as plt

root = './output/dynerf/'
folder = 'flame_salmon/'

focal_word = 'covloss1'
bench_marks = ['covloss','wavelevel','pseudodepth1','tune','l1001norms','l10001norms','tv001norms', 'tv0005norms', 'w2norms','w4norms','1norms','01norms','001norms','0005norms','0025norms','005norms','sigW1']


# ,'sigW4', 'sigW5', 'sigW6', 'sigW7','sigW8','sigW9','rigid1','rigid3']


non_words = [] # ['tv', 'HW', 'Shared','NoCos','4-D','OpacEm', 'Bez','Seq',  'Fix','ts','sep', 'tv' 'Feat', 
            #  'temb', 'Test', 'test','reset', 'Sep', 'Activation', 'd0', 'rotation', 'LR', 'Reg', '12k']

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

    