# Fault Diagnosis in New Wind Turbines based on Knowledge from Existing Wind Turbines by Generative Domain Adaptation

Official implementation of our research article, available on [arXiv](https://arxiv.org/). 

---

### Usage Example (Tutorial)

This repository provides no SCADA datasets for running the experiments. For illustrative purposes only of our implementation, we provide a dataset structure in /datasets/. In our code, we consider a wind turbine from "farmS" the source domain, and a wind turbine from "farmT" the target domain. Each farm consists here of one WT, with ID 1 (WT_1). However, the SCADA measurements (.csv files) consist of only 1 exemplary row of data, *which is not suitable for running the scripts*. They only give an example for the data structure and the date & value formatting to be followed. The "meta data" (META.csv) containing respective rated power and wind speed information is also to be used as a formatting guide. The directory structure, data paths, SCADA measurement formatting, and corresponding meta data is meant to be replaced with your proprietary datasets.

Once data has been replaced, consider the following usage examples.

#### (1) **representative**: Train an NBM (without data scarcity) on the source and target domain each:

Run 

```
[python] train_NBM.py -SITE_NAME="farmS" -WT_ID=1 [opt]
```

(We recommend using CUDA-supported GPU for training procedures; specify through adding e.g., ```-CUDA_IDX=0``` )

to train a source WT NBM. For the target WT, run

```
[python] train_NBM.py -SITE_NAME="farmT" -WT_ID=1 -CUDA_IDX=0 [opt]
```

The script will automatically save the NBMs in the /saves/NBM/ directory.


#### (2) **scarce scenario**: Train an NBM for the target domain with a data scarcity scenario (here: 2 weeks of training data)

Run 

```
[python] train_NBM.py -SITE_NAME="farmT" -WT_ID=1 -SCARCITY="2w" -CUDA_IDX=0 [opt]
```

The script will automatically save the NBM in another /saves/NBM/ directory.

#### (3) **fine-tuning**: Fine-tune the representative source NBM using scarce target WT training data: 

Run 

```
[python] train_finetune.py -SITE_NAME_S="farmS" -WT_ID_S=1 -SITE_NAME_T="farmT" -WT_ID_T=1 -SCARCITY="2w" -CUDA_IDX=0  [opt]
```

to fine-tune the _pretrained_ NBM of farmS (WT 1) on the scarce target WT data. The fine-tuned model is saved in the /saves/finetune/ directory.

#### (4) **domain mapping**: Train a domain mapping network with the source WT data and scarce target WT data

Run 

```
[python] train_domain_mapping.py -SITE_NAME_S="farmS" -WT_ID_S=1 -SITE_NAME_T="farmT" -WT_ID_T=1 -SCARCITY="2w" -CUDA_IDX=0 [opt]
```

The domain mapping training script will train two generators mapping from the source domain to the target domain and vice versa. The generators are saved in the /saves/mapping/ directory.

#### (5) **evaluation/comparison**: Run the evaluation script to compare the obtained test set anomaly scores of the scarce NBM, the fine-tuned NBM, and the domain mapping approach to the ground truth, in our case the anomaly scores from the representative target WT NBM.

Run 

```
[python] evaluate_models.py -SITE_NAME_S="farmS" -WT_ID_S=1 -SITE_NAME_T="farmT" -WT_ID_T=1 -SCARCITY="2w" -CUDA_IDX=0
```

The metrics, including the F1-score, will be stored in a .csv file in the /results/ directory.

-----------------
### Citation
To cite our work:
```
@article{WTDomainMapping,
  title={Fault Diagnosis in New Wind Turbines based on Knowledge from Existing Wind Turbines by Generative Domain Adaptation},
  author={Stefan Jonas, Angela Meyer},
  year={2025}
}
```