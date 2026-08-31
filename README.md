# TCSF

TCSF: A Tri-Level Cross-State Fusion Network with Vision Mamba and Adaptive Reliability Learning for SAR-Optical Flood Mapping

TCSF is a multimodal deep-learning framework for flood segmentation using co-registered Sentinel-1 SAR and Sentinel-2 optical imagery. The model combines reliability-aware multimodal learning, dual Vision Mamba encoders, bidirectional Cross-State Fusion, controlled cross-scale state propagation, and Adaptive Decision Fusion to improve flood delineation across heterogeneous scenes.

The implementation is developed in PyTorch and evaluated on SEN1FLOODS11, SEN12MS, and DEEPFLOOD. Experiments were conducted using an NVIDIA RTX A4000 GPU with 16 GB memory.

Paper status: Manuscript/preprint. A publication link and final bibliographic citation will be added after publication.

## Contents
1. [Introduction](#introduction)
2. [Key Highlights](#keyhighlights)
3. [Dependencies](#dependencies)
4. [Train](#train)
5. [Test](#test)
6. [Results](#results)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

## Introduction

Flood mapping benefits from combining Sentinel-1 SAR and Sentinel-2 optical imagery because the two modalities provide complementary information. SAR offers all-weather, day-and-night observations but is affected by speckle noise and complex scattering, while optical imagery provides rich spectral information but can be degraded by clouds, haze, and shadows.

TCSF — **Tri-Level Cross-State Fusion Network** — is a reliability-aware multimodal framework that maintains separate SAR and optical feature streams rather than directly concatenating the inputs. It combines **Modality Reliability Estimation, dual Vision Mamba encoders, Cross-State Fusion, controlled tri-level state propagation, and Adaptive Decision Fusion** to learn complementary multimodal representations for flood segmentation.

The model uses Sentinel-1 VV/VH and all 13 Sentinel-2 spectral bands.

### Overall TCSF Framework

<p align="center">
  <img src="./outputs/publication/architecture_v2/tcsf_v31_overall_architecture.png" width="100%" alt="TCSF Architecture">
</p>

TCSF uses four reliability-guided Cross-State Fusion stages connected through three controlled cross-scale transitions. SAR, optical, and fused states are progressively propagated across encoder scales and decoded through separate prediction branches. Adaptive Decision Fusion then combines the three predictions using learned pixel-wise weights.

### Cross-State Fusion

<p align="center">
  <img src="./outputs/publication/architecture_v2/tcsf_v31_csf_block.png" width="90%" alt="TCSF Cross-State Fusion">
</p>

Cross-State Fusion enables bidirectional interaction between SAR and optical features while using learned modality reliability to control cross-modal information exchange. Three controlled transitions connect the four fusion stages:

**CSF1 → T1 → CSF2 → T2 → CSF3 → T3 → CSF4**

Each transition propagates SAR, optical, and fused states using bounded learnable residual coefficients.

## Key Highlights

* **Reliability-Aware Fusion:** Learns pixel-level SAR and optical confidence maps to reduce the influence of unreliable observations.

* **Dual Vision Mamba Encoders:** Preserve modality-specific information while capturing long-range spatial dependencies.

* **Tri-Level Cross-State Fusion:** Performs reliability-guided multimodal interaction at four scales with controlled state propagation across three transitions.

* **Adaptive Decision Fusion:** Combines SAR-only, optical-only, and fused predictions using learned pixel-wise weights.

* **Multi-Objective Training:** Uses BCE, Dice, boundary, and consistency losses for accurate flood-region and boundary prediction.

* **Multi-Dataset Evaluation:** Evaluated on SEN1FLOODS11, SEN12MS, and DEEPFLOOD against CNN-, Transformer-, and Vision Mamba-based baselines.

## Dependencies
* Python 3.1
* PyTorch >= 1.1.0
* CUDA 12.2
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Train
### Prepare training data 

1. Download DEEPFLOOD Dataset, which includes co-registered Sentinel-1 SAR (VV, VH) and Sentinel-2 optical imagery, along with UAV references and auxiliary layers (NDWI, slope, DTM, flood masks). from [DEEPFLOOD dataset](https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339).

2. Download SEN1FLOODS11 Dataset, from [SEN1FLOODS11 dataset](https://github.com/cloudtostreet/Sen1Floods11)

3. Download SEN12MS Dataset, from [SEN12MS dataset](https://mediatum.ub.tum.de/1474000)
Input Preparation

All input images and labels are resized to:

256 × 256 pixels

Training augmentations must be applied consistently to SAR, optical, and label tensors so that their spatial correspondence is preserved.

For qualitative visualization, selected channels may be displayed for interpretability, such as Sentinel-1 VV and Sentinel-2 RGB. Training and evaluation use the complete multimodal input configuration.

Training

The main TCSF model is trained using train.py.

Example

python train.py \
    --config configs/acsf_base.yaml

The principal training protocol in the manuscript uses:

Epochs: 20
Seeds: 42, 123, 2026

A separate 200-epoch experiment is used for convergence and long-term optimization analysis.

Training Objective

The complete loss is composed of:

Ltotal =
    1.0 × LBCE
  + 1.0 × LDice
  + 0.2 × LBoundary
  + 0.1 × LConsistency

The best checkpoint is selected according to validation IoU.

Testing

Use test.py to evaluate a trained TCSF checkpoint.

python test.py \
    --config configs/acsf_base.yaml \
    --checkpoint outputs/checkpoints/TCSF_v31_seed2026/best_model.pth

The evaluation reports:

IoU
Dice / F1
Precision
Recall
Pixel Accuracy

The binary segmentation threshold is:

0.5

Checkpoints are not included in this GitHub repository and must be generated locally or obtained separately.

Inference

Use infer.py to generate predictions and qualitative visualizations.

python infer.py \
    --config configs/acsf_base.yaml \
    --checkpoint outputs/checkpoints/TCSF_v31_seed2026/best_model.pth \
    --save_dir outputs/predictions/TCSF_v31_seed2026

The inference pipeline can export:

Final flood prediction

SAR-only prediction

Optical-only prediction

SAR reliability map

Optical reliability map

Decision weights

Multimodal comparison figures

Results

Quantitative Comparison

TCSF was compared against U-Net, DeepLabV3+, SegFormer-B0, Swin-UNet, and Vision Mamba.

SEN1FLOODS11

Model

Params. (M)

IoU

Dice

Precision

Recall

Pixel Acc.

DeepLabV3+

40.38

0.4522

0.5585

0.6340

0.5976

0.9514

SegFormer-B0

3.73

0.4963

0.6009

0.6385

0.6084

0.9552

Swin-UNet

6.61

0.5187

0.6254

0.6412

0.6232

0.9586

U-Net

7.85

0.5236

0.6331

0.6498

0.6635

0.9613

Vision Mamba

5.57

0.5280

0.6320

0.6781

0.6948

0.9619

TCSF (Ours)

11.02

0.5803

0.6833

0.7653

0.6979

0.9680

SEN12MS

Model

Params. (M)

IoU

Dice

Precision

Recall

Pixel Acc.

DeepLabV3+

40.38

0.4387

0.5441

0.6215

0.5819

0.9476

SegFormer-B0

3.73

0.4824

0.5898

0.6267

0.5963

0.9518

Swin-UNet

6.61

0.5049

0.6137

0.6351

0.6104

0.9557

U-Net

7.85

0.5116

0.6208

0.6429

0.6467

0.9589

Vision Mamba

5.57

0.5198

0.6259

0.6672

0.6751

0.9596

TCSF (Ours)

11.02

0.5689

0.6764

0.7488

0.6895

0.9661

DEEPFLOOD

Model

Params. (M)

IoU

Dice

Precision

Recall

Pixel Acc.

DeepLabV3+

40.38

0.4678

0.5729

0.6473

0.6138

0.9536

SegFormer-B0

3.73

0.5095

0.6153

0.6516

0.6241

0.9571

Swin-UNet

6.61

0.5314

0.6376

0.6597

0.6412

0.9605

U-Net

7.85

0.5382

0.6451

0.6689

0.6724

0.9628

Vision Mamba

5.57

0.5467

0.6518

0.6935

0.7012

0.9637

TCSF (Ours)

11.02

0.5968

0.7019

0.7794

0.7146

0.9708

Overall Dataset Performance

Dataset

IoU

Dice

SEN1FLOODS11

0.5803

0.6833

SEN12MS

0.5689

0.6764

DEEPFLOOD

0.5968

0.7019

Computational Efficiency

For SEN1FLOODS11, TCSF achieves:

Trainable Parameters: 11.02 M
Macro IoU:            0.5803
Inference Latency:    77.1 ms/image

The model introduces additional computation relative to lightweight baselines, but provides a favorable accuracy-complexity trade-off and remains faster than DeepLabV3+ in the reported benchmark.

Ablation Study

The manuscript evaluates the contribution of the major TCSF components.

Reliability Estimation

Configuration

Macro IoU

Macro Dice

TCSF w/o Reliability

0.5576

0.6612

TCSF with Reliability

0.5612

0.6685

Cross-State Fusion

Configuration

Macro IoU

Macro Dice

TCSF w/o CSF

0.5489

0.6534

TCSF with CSF

0.5583

0.6596

Controlled Tri-Level Propagation

Configuration

Macro IoU

Macro Dice

TCSF w/o Tri-Level Propagation

0.4521

0.5567

TCSF with Tri-Level Propagation

0.5797

0.6833

Adaptive Decision Fusion

Configuration

Macro IoU

Macro Dice

TCSF w/o ADF

0.5594

0.6638

TCSF with ADF

0.5623

0.6693

The controlled tri-level propagation mechanism produces the largest performance change among the reported component ablations, supporting the importance of progressive cross-scale state transfer.

Baseline Models

The repository also contains training, testing, and inference implementations for the evaluated baselines:

U-Net
DeepLabV3+
SegFormer-B0
Swin-UNet
Vision Mamba

Example baseline scripts include:

train_unet.py
test_unet.py
infer_unet.py

train_deeplabv3plus.py
test_deeplabv3plus.py
infer_deeplabv3plus.py

train_segformer.py
test_segformer.py
infer_segformer.py

train_swin_unet.py
test_swin_unet.py
infer_swin_unet.py

train_vision_mamba.py
test_vision_mamba.py
infer_vision_mamba.py

Citation

If you find this repository useful in your research, please cite the TCSF manuscript.

@unpublished{talreja2026tcsf,
  author = {Jagrati Talreja and Tewodros Syum Gebre and Leila Hashemi Beni},
  title  = {TCSF: A Tri-Level Cross-State Fusion Network with Vision Mamba and Adaptive Reliability Learning for SAR-Optical Flood Mapping},
  year   = {2026},
  note   = {Preprint submitted to Elsevier}
}

The citation will be updated when the final publication information becomes available.

Acknowledgements

This research was conducted at the College of Science and Technology, North Carolina A&T State University.

The authors acknowledge the developers and maintainers of the SEN1FLOODS11, SEN12MS, and DEEPFLOOD datasets, as well as the open-source PyTorch and remote-sensing research communities that support reproducible multimodal Earth-observation research.

Authors

Jagrati Talreja
College of Science and Technology, North Carolina A&T State University

Tewodros Syum Gebre
College of Science and Technology, North Carolina A&T State University

Leila Hashemi Beni
College of Science and Technology, North Carolina A&T State University
Institute for Water, Environment and Health, United Nations University

License

Please refer to the repository license for terms of use. If no license has yet been added, the code remains subject to the copyright of the authors.
