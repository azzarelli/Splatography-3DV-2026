import os
import json
import matplotlib.pyplot as plt

root = '/home/barry/Desktop/mywork/GS_Research/output/dynerf/'
folder = 'flame_steak/'

focal_word = 'Shared'
non_words = []

root =  os.path.join(root, folder)

runs = os.listdir(root)

x = {}
y_psnr = []
y_ssim = []
names = []

for run in runs:
    directory = os.path.join(root, run)
    directory = os.path.join(directory, 'active_results')
    
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

for result, name in zip(y_psnr, names):
    check = 0
    for n in non_words:
        if n in name:
            check += 1
    if (focal_word in name or 'bench' in name) and check == 0:
        print(name)
        print(result)
        if 'bench' in name:
            axes[0].plot(result, label=f"{name}", color='black')  
        else: axes[0].plot(result, label=f"{name}")  # Add a label for each set
# Add a label for each set
axes[0].set_title("PSNR")
axes[0].legend()  # Show the legend
axes[0].grid(True)  # Add a grid

# Plot y_ssim results
for result, name in zip(y_ssim, names):
    check = 0
    for n in non_words:
        if n in name:
            check += 1
    if (focal_word in name or 'bench' in name) and check == 0:
        if 'bench' in name:         axes[1].plot(result, label=f"{name}", color='black')  # Add a label for each set
        else: axes[1].plot(result, label=f"{name}")  # Add a label for each set
axes[1].set_title("SSIM Results")
axes[1].legend()  # Show the legend
axes[1].grid(True)  # Add a grid

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

    