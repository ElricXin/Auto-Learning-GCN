# Auto-Learning-GCN
This repo is the official implementation for Auto-Learning-GCN: <u>An Ingenious Framework for Skeleton-based Action Recognition<u>

## architecture of Auto-Learning-GCN




# Prerequisites

+ Python >= 3.6

+ PyTorch >= 1.1.0

+ PyYAML, tqdm, tensorboardX

# Data Preparation

### Download datasets

**There are 4 datasets to download:**
+ NTU RGB+D 60 Skeleton
+ NTU RGB+D 120 Skeleton
+ NW-UCLA
+ UAV-Human

**NTU RGB+D 60 and 120**

1. Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:  
    i. ```nturgbd_skeletons_s001_to_s017.zip``` (NTU RGB+D 60)  
    ii. ```nturgbd_skeletons_s018_to_s032.zip``` (NTU RGB+D 120)  
    iii. Extract above files to ```./data/nturgbd_raw```  

**UAV-Human**

1. Download dataset from here: https://sutdcv.github.io/uav-human-web/
2. Move ```Skeleton``` to ```./data/UAV-Human```

**NW-UCLA**

1. Download dataset from [here](https://drive.google.com/file/d/1wWhgqMEQlrCKcJHu6W72Zk_iloS7_JJw/view?usp=share_link)
2. Move ```all_sqe``` to ```./data/NW-UCLA```



### NTU Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:
~~~
- data/
  - UAV-Human/
    - Skeleton
      ... # raw data of UAV-Human
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
~~~

#### Generating Data

+ Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
~~~
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
~~~

#### Generate Data:

+ Generate UAV-Human dataset:
~~~
 cd ./data/uav/Skeleton
 # Updata statistics.py
 python updata_statistics.py
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
~~~
The pre-processed UAV data can be referred [here](https://drive.google.com/drive/my-drive) and details can be found [Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer)

    
# Training & Testing
### Training
+ Change the config file depending on what you want.
~~~
    # Example: training AL-GCN on NTU RGB+D cross subject with GPU 0
    python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/algcn --device 0
    # Example: training provided baseline on NTU RGB+D cross subject
    python main.py --config config/nturgbd-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu/csub/baseline --     device 0
~~~
+ To train model on NTU RGB+D 60/120 with bone or motion modalities, setting ```bone``` or ```vel``` arguments in the config file ```default.yaml``` or in the command line.
~~~
    # Example: training AL-GCN on NTU RGB+D 120 cross subject under bone modality
    python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.algcn_ntu.py --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/algcn_bone --device 0
~~~
+ To train model on NW-UCLA with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/ucla/default.yaml --model model.algcn_ucla.py --work-dir work_dir/ucla/algcn_ucla --device 0
~~~
+ To train model on UAV-Human with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/uav/default.yaml --model model.algcn_uav.py --work-dir work_dir/uav/algcn_uav --device 0
~~~

### Testing

+ To test the trained models saved in <work_dir>, run the following command:  

~~~
    python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --         device 0
~~~

+ To ensemble the results of different modalities, run  

~~~
    # Example: ensemble four modalities of AL-GCN on NTU RGB+D cross subject
    python ensemble.py --dataset ntu/xsub --joint-dir work_dir/ntu/csub/algcn --bone-dir work_dir/ntu/csub/algcn_bone --joint-motion-dir     work_dir/ntu120/csub/algcn_motion --bone-motion-dir work_dir/ntu/csub/algcn_bone_motion
~~~

### Pretrained model
We are diligently curating relevant data and models, and our pre-trained model will soon be made public. If you are interested, please stay tuned for our upcoming work.
    
## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).


Thanks to the original authors for their work!  


