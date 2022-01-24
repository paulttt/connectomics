# connectomics

This Code was written within a research internship at Lichtman and Engert Labs at the Department of Molecular and Cellular Biology at Harvard University. 
Parts of this work will be published in a joint publication within a neuroscientific journal.

At first, this work benchmarks three state-of-the-art nuclei segmentation models on four datasets acquired with 2P, X-ray-based Micro Computed Tomography (μCT), and EM imaging from a larvae zebrafish brain. A new metric is proposed called Average Distance between Centroids (ADC) to measure the centroid alignment quality of algorithmic predictions compared to ground truth data.
The code for this project can be found in the subfolder CellAnalysis. Moreover the package [mAP_3Dvolume](https://github.com/ygCoconut/mAP_3Dvolume) was forked and modified for segmentation of 2D images and usage within this project.

In addition, a dataset consisting of neuronal activity data from GCaMP7f expressing neurons has been used as input for different time series classification models: For the first time, it has been tried to learn excitatory and inhibitory labels retrieved from gad1b:DsRed expressing cells using 2P recordings from zebrafish brain.

## CellAnalysis

This toolbox supports the evaluation and benchmarking of segmentation masks of neuronal cell bodies for 3D and 2D images. Most of the high level easy-to-use classes and functions are implemented in the ```CellAnalysis/eval.py``` file. 
The most easy-to-use and straightforward way to benchmark a set of different prediction outcomes from different models is to use the following code:
```
from CellAnalysis.eval import benchmark
import os

path = os.path.dirname(os.path.abspath(""))+"/"
file_root = path + 'example_data/EM Data'
benchmark = benchmark(file_root, resolution=(0.6, 0.6, 0.6))
```
Here we pass a folder to the function, where all the relevant files can be found in its subdirectories. It is really important that the user follows the exact same file/directory structure as mentioned below. The names of the files can be chosen arbitrarly by the user, it is just important that the names are consisent across the different volumes/roi as they get read in in a sorted fashion.
```
EM\ Data
    ├── gt
    │   ├── ROI_1.tif
    │   ├── ROI_2.tif
    │   └── ROI_3.tif
    └── prediction
        ├── Cellpose
        │   ├── ROI_1.tif
        │   ├── ROI_2.tif
        │   └── ROI_3.tif
        ├── PyTC
        │   ├── ROI_1.tif
        │   ├── ROI_2.tif
        │   └── ROI_3.tif
        └── Stardist
            ├── ROI_1.tif
            ├── ROI_2.tif
            └── ROI_3.tif
```
You can vary the amount of folders in the ```prediction``` folder, it is just important that the respective files that come from the same region of interest (roi) have the same naming scheme. Moreover, the program will associate the names of the directories that are contained in the ```prediction``` folder automatically to the data containing it and reference it to the visualized results from the class.

If we call the function ```summarize()``` on the above instantiated class instance ```benchmark```, we could for example quickly visualize the calculated metrics from those three segmentation predictions and see how good they perform in various tasks:
```
benchmark.summarize(title='Electron Microscopy - Neuronal Cell Body Segmentation Results', save_to_file='EM_Results', figsize=(25, 9), file_type='pdf')
```
We will get the following results then:
![Summary of EM Segmentation Evaluation](https://github.com/paulttt/connectomics/blob/main/figures/EM_Results.png)

