import numpy as np
from dataset import load_data_3D, get_pairs, split_pairs
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data/"

# dataset stats
pairs = get_pairs(root)
patients = {mr.split("/")[-1].split("_")[1] for mr, _ in pairs}
print("Scans:", len(pairs), "Patients:", len(patients))
print("MR shapes:", {nib.load(mr).shape for mr, _ in pairs}) # >1 shape means load_data_3d clips or pads

print(sorted([nib.load(mr).shape for mr, _ in pairs]))

for name, split in split_pairs(pairs).items():
    print(name, len(split), "scans")

image = load_data_3D([root + "semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz"])
label = load_data_3D([root + "semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz"], dtype=np.uint8)

print("img:", image.shape, image.dtype, "min-max:", image.min(), image.max())
print("label:", label.shape, label.dtype, "types, counts:", np.unique(label, return_counts=True))

plt.subplot(121); plt.imshow(image[0, :, :, 64], cmap="gray")
names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
cmap = plt.get_cmap("tab10", len(names))
plt.subplot(122); plt.imshow(label[0, :, :, 64], cmap=cmap, vmin=0, vmax=len(names) - 1, interpolation="nearest")
plt.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="upper left", bbox_to_anchor=(1, 1))
plt.show()