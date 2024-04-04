## <div align="center">Preface</div>
This project is based on the code of [yolov5-6.2 version](https://github.com/ultralytics/yolov5/releases/tag/v6.2) and performs pruning. Compared to the original yolov5-6.2, this project integrates all the files required for pruning into the [prune_tools](prune_tools) folder, with the rest remaining the same.
This project takes mask-wearing detection as an example to demonstrate how **yolov5n** is pruned. The same operations apply to other series, which will not be repeated here.
  ```shell
 prune_tools/
 ├── finetune.py            # For fine-tuning after pruning
 ├── prune.py               # Start pruning the yolo model
 ├── pruned_common.py       # Module for the model after pruning
 ├── train_sparity.py       # Sparse training
 ├── yolo_pruned.py         # Build the model after pruning
```

The pruning method of this project comes from [Learning Efficient Convolutional Networks Through Network Slimming](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html), which prunes based on the size of the γ coefficients of the BN layer.

1. First, perform sparse training to press the γ coefficients of the BN layer to smaller values.

2. Set the pruning ratio and prune the channels corresponding to the overly small γ coefficients of the BN layer.

3. Fine-tune.


## <div align="center">Instructions for Use</div>
<details open>
<summary>Environment Installation</summary>

You can refer to my [requirements.txt](requirements.txt) for downloading, **Python=3.9**.

```bash
pip install -r requirements.txt  # install
```

</details>


<details open>
<summary>Basic Training</summary>

The training method can refer to my blog post [Yolov5 Mask Wearing Real-time Detection Project (opencv+python inference)](https://blog.csdn.net/weixin_43490422/article/details/127148825?spm=1001.2014.3001.5501)

1. Download the dataset from the Baidu Cloud link: [mask_yolo](https://pan.baidu.com/s/1l1-lBb_znAYcFmRVDj1IfQ?pwd=glxz)

2. The dataset configuration file [mask.yaml](data/mask.yaml) and model configuration file [yolov5n_mask.yaml](models/yolov5n_mask.yaml) have been included in the project.
   
3. Modify the path in the dataset configuration file [mask.yaml](data/mask.yaml) to your dataset's location

```bash
path: ../datasets/mask_yolo  # Modify to your dataset path
```

4. Start training
```bash
python train.py --data data/mask.yaml --cfg models/yolov5n_mask.yaml --weights yolov5n.pt --batch-size 64 --epochs 200 --imgsz 320
```
**Training results**:

| Class   | Images | Instances | P     |    R  | mAP50 | mAP50-95 | params<br><sup>(M) | GFLOPs<br><sup>@320 |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| all     | 1839   | 3060      | 0.953|  0.913| 0.949  | 0.657   | 1.77                | 4.2           
| face     | 1839   | 2024      | 0.955|  0.899| 0.935  | 0.633   | --                | --          
| face_mask     | 1839   | 1036      | 0.952|  0.928| 0.963  | 0.681  | --                | --          


</details>


<details open>
<summary>Sparse Training</summary>

```bash
python prune_tools/train_sparity.py --st --sr 0.0005 --weights runs/train/exp3/weights/best.pt --cfg models/yolov5n_mask.yaml --data data/mask.yaml --batch-size 64 --epochs 100 --imgsz 320
```

Where:
1. **weights**: Path to the best weight file obtained from basic training

2. **st**: Whether to start sparse training

3. **sr**: Sparsity factor, the larger it is set, the more γ factors close to 0, which can be adjusted based on the histogram effect and mAP performance

The distribution of the γ weights of the BN layer after sparsity can be viewed with the following

 command, then manually enter http://localhost:6006/ in the browser.
```bash
tensorboard --logdir=runs/train
```

The histogram of the γ factors after sparsity is shown below:
![image](readmeImg/tensorboard.png)

**Sparse results**:

| Class   | Images | Instances | P     |    R  | mAP50 | mAP50-95 | params<br><sup>(M) | GFLOPs<br><sup>@320 |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| all     | 1839   | 3060      | 0.952|  0.908| 0.949  | 0.642   | 1.77                | 4.2           
| face     | 1839   | 2024      | 0.944|  0.888| 0.936  | 0.622   | --                | --          
| face_mask     | 1839   | 1036      | 0.961|  0.927| 0.961  | 0.663  | --                | --          



</details>



<details open>
<summary>Pruning</summary>


```bash
python prune_tools/prune.py --weights runs/train/exp4/weights/best.pt --percent 0.4 --cfg models/yolov5n_mask.yaml --data data/mask.yaml --batch-size 64 --imgsz 320
```
Where:
1. **weights**: Path to the best weight file obtained from sparse training

2. **percent**: Specifies the percentage of parameters to be pruned


**Pruning results**:

| Class   | Images | Instances | P     |    R  | mAP50 | mAP50-95 | params<br><sup>(M) | GFLOPs<br><sup>@320 |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| all     | 1839   | 3060      | 0.375|  0.536| 0.311  | 0.0876   | 0.82                | 3.1            

It can be seen that the accuracy significantly decreases after pruning, requiring fine-tuning operations.

</details>


<details open>
<summary>Fine-tuning</summary>

```bash
python prune_tools/finetune.py --data data/mask.yaml  --weights runs/val/exp/pruned_model.pt --batch-size 64 --epochs 100 --imgsz 320
```

Where:
1. **weights**: Refers to the weight after pruning, placed in the runs/val folder

**Fine-tuning results**:

| Class   | Images | Instances | P     |    R  | mAP50 | mAP50-95 | params<br><sup>(M) | GFLOPs<br><sup>@320 |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| all     | 1839   | 3060      | 0.952|  0.918| 0.945  | 0.652   | 0.82                | 3.1           
| face     | 1839   | 2024      | 0.948|  0.907| 0.933  | 0.63   | --                | --          
| face_mask     | 1839   | 1036      | 0.952|  0.928| 0.957  | 0.673  | --                | --          



</details>