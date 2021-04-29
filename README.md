# connectomics


## CellAnalysis

This toolbox supports the evaluation and benchmarking of segmentation masks of neuronal cell bodies for 3D and 2D images. Most of the high level easy-to-use classes and functions are implemented in the ```CellAnalysis/eval.py``` file. 
The most easy-to-use and straightforward way to benchmark a set of different prediction outcomes from different models is to use the following code:
```
from CellAnalysis.eval import benchmark
import os

path = os.path.dirname(os.path.abspath(""))+"/"
file_root = path + 'example_data/EM Data'
benchmark = benchmark(file_root, resolution=(0.51, 0.51, 0.51))
```
Here we pass a folder to the function, where all the relevant files can be found in its subdirectories. It is really important that the user follows the exact same file/directory structure as mentioned below:
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
benchmark.summarize(title='Electron Microscopy - Neuronal Cell Body Segmentation Results', save_to_file='EM_Results', figsize=(20, 9))
```
We will get the following results then:
![Summary of EM Segmentation Evaluation](https://github.com/paulttt/connectomics/blob/main/examples/EM_Results.png?raw=true)

