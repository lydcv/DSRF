# DSRF: Few-Shot PCB Surface Defect Detection via Dynamic Selective Regulation Fusion
PyTorch implementation for the paper: Few-Shot PCB Surface Defect Detection via Dynamic Selective Regulation Fusion 
![](https://github.com/lydcv/DSRF/raw/main//DSRF.png) 
Our code is based on [mmfewshot](https://github.com/open-mmlab/mmfewshot).
## Requirements
* python == 3.8.19
* torch == 1.8.1
* torchvision == 0.8.0
* mmcv-full == 1.3.12 
* mmdet == 2.16.0
* mmcls == 0.15.0
* mmfewshot == 0.1.0
* cuda == 11.1
## Train and Test
### Train
Base trianing:
```bash
# single gpu
python train.py --config=configs/dsrf/deeppcb/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_base-training.py 

# multi gpu
bash dist_train.sh --config=configs/dsrf/deeppcb/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_base-training.py 2
```
Fine-tuning:
```bash
# single gpu
pcb_config_dir="configs/dsrf/deeppcb/"
for shot in 1 2 3 5 10; do
    config_path="${pcb_config_dir}/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_${shot}shot-fine-tuning.py"
    echo "$config_path"
    python train.py --config="$config_path" 
done

# multi gpu
pcb_config_dir="configs/dsrf/deeppcb/"
for shot in 1 2 3 5 10; do
    config_path="${pcb_config_dir}/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_${shot}shot-fine-tuning.py"
    echo "$config_path"
    bash dist_train.sh --config="$config_path"  2
done
```
### Test
```bash
# single gpu
python test.py --config=configs/dsrf/deeppcb/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_2shot-fine-tuning.py --checkpoint=work_dirs/dsrf_r101_c4_8xb4_pcb-split1_2shot-fine-tuning/iter_800.pth --eval mAP

# multi gpus
bash dist_test.sh --config=configs/dsrf/deeppcb/dsrf_split1/dsrf_r101_c4_8xb4_pcb-split1_1shot-fine-tuning.py --checkpoint=work_dirs/dsrf_r101_c4_8xb4_pcb-split1_1shot-fine-tuning/iter_300.pth 2 --eval mAP
```
