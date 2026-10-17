import numpy as np
from dataset import load_data_3D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data/"
image = load_data_3D([root + "semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz"])
label = load_data_3D([root + "semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz"], dtype=np.uint8)

print(image.shape, image.dtype, image.min(), image.max())
print(label.shape, label.dtype, np.unique(label, return_counts=True))

plt.subplot(121); plt.imshow(image[0, :, :, 64], cmap="gray")
names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
cmap = plt.get_cmap("tab10", len(names))
plt.subplot(122); plt.imshow(label[0, :, :, 64], cmap=cmap, vmin=0, vmax=len(names) - 1, interpolation="nearest")
plt.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="upper left", bbox_to_anchor=(1, 1))
plt.show()