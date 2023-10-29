# Retrieval-Augmented Multiple Instance Learning (RAM-MIL)

This repository contains preprint and code for Retrieval-Augmented Multiple Instance Learning (NeurIPS 2023).

## ✓ Requirements

Use the environment configuration the same as CLAM:

```setup
conda env create -n clam -f CLAM/clam.yaml

pip install pot
pip install geomloss
```
## ✓ Training
##  CLAM Pretraining.
1. Get into the clam directory.
```bash
cd CLAM/
```

2. Split the data into 10-folds, then save the splited data in the following format as in `splits/xxx/splits_x.csv`.
```bash
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --label_frac 0.75 --k 10
```

3. Training the clam_sb model.
```bash
python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 0.75 --exp_code task_1_tumor_vs_normal_CLAM_75 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir data_root_dir --results_dir result
```

4. Save the slide-level feature and attention scores.
>📋  The pre-trained attention scores can be download: [`c16_attention`](ot_retrieval/c16_attention/) 
```bash
python save_slides.py
```



## Start Neighbor Retrieval
1. Get into the ot_retrieval directory.. 
```bash
cd ot_retrieval
```

2. Save the top 10% or 20% patch features and attention scores.
>📋 To increase the calculation speed, it is better to save the features in advance.
```bash
python attention.ipynb

```
3. Modify the file list.

`emb_c16_sort.txt` and `emb_c17_sort.txt` are sorted by the number of patches.

4. Save optimal transport loss.
>📋 Due to the large size of the tensor, there is a risk of memory explosion during the computation process. 
Therefore, it is recommended to consider splitting the data and allocating GPUs for parallel computation.

```bash

python opt.pyz --attn_a PATH_TO_ATTENTION_A --attn_b PATH_TO_ATTENTION_B --feat_a PATH_TO_FEATURE_A --feat_b PATH_TO_FEATURE_B --pct 1.0 --save_path PATH_TO_SAVE
```

5. Retrieve nearest neighbors.

```bash
python retrieve_neighbor.py
```


## Classifier Training
1. Get into the ot_retrieval directory.. 
```bash
cd Classifier
```

2. Modify the corresponding neighbor file `loss_matrix_1616_20att_xx.json`.
```bash
cd datasets1/dataset_generic.py
```
```python
in class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset)

# retrieval in-domain:
indomain_nebs/*.json

# retrieval in domain and out-of-domain:
inout_nebs/*.json

# retrieval out-of-domain:
out_nebs/*.json
```

3. Merge Function.
```bash
cd models/model_clam.py
## CLAM_SB: simple addition
## CLAM_SB_ADD: convex combination

```

4. Classifier Training.
```bash
CUDA_VISIBLE_DEVICES=0 nohup python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 0.75 --exp_code task_1_tumor_vs_normal_CLAM_75 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb_add --log_data --data_root_dir slide_feature_dir --results_dir ./result_add --reg 1e-4 >  result-add.log &
```

## ✓ Evaluation

>📋 To evaluate my model under in-domain setting, run:

```eval
python -u eval_c16.py --drop_out --k 10 --models_exp_code task_1_tumor_vs_normal_CLAM_75_s1 --save_exp_code result_add --task task_1_tumor_vs_normal --model_type clam_sb_add --results_dir ./result-add --data_root_dir c16_slide_feature_dir --splits_dir CLAM/splits/task_1_tumor_vs_normal_75
```

>📋 To evaluate my model under out-of-domain setting, run:
```bash
# Modify the corresponding neighbor file loss_matrix_1616_20att_xx.json to loss_matrix_1716_20att_xx.json/loss_matrix_1717_20att_xx.json.
python -u eval_c17.py --drop_out --k 10 --models_exp_code task_1_tumor_vs_normal_CLAM_75_s1 --save_exp_code c17_result --task task_1_tumor_vs_normal --model_type clam_sb_add --results_dir ./result-add --data_root_dir c17_slide_feature_dir --splits_dir CLAM/c17_splits/task_1_tumor_vs_normal_75
```


# License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.


## Reference

Please cite our paper if you use the core code of RAM-MIL. 

Yufei, Cui, et al. "Retrieval-Augmented Multiple Instance Learning." Thirty-seventh Conference on Neural Information Processing Systems, 2023.

```
@inproceedings{yufei2023retrieval,
  title={Retrieval-Augmented Multiple Instance Learning},
  author={Yufei, Cui and Liu, Ziquan and Chen, Yixin and Lu, Yuchen and Yu, Xinyue and Liu, Xue and Kuo, Tei-Wei and Rodrigues, Miguel and Xue, Chun Jason and Chan, Antoni B},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
