# SGAFusion：Semantic-Guided Adaptive Fusion for Infrared-Visible Images Under Degradation Conditions

> This repository provides a non-official and simplified version of our implementation, intended solely for academic communication and preliminary understanding.  
> To protect the originality of our work and ensure compliance with data and intellectual property considerations, the **complete codebase will be released after the paper is officially published**.
> We appreciate your understanding and support.

## 🔹 Framework Overview

The structure of **SGAFusion** is illustrated in the figure below:


![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Algorithm%20Framework.jpg)


*Fig. 1: Overall Architecture of SGAFusion. Semantic information representing degradation types is automatically generated from the visible image via the CLIP model, and used as the query (Q) in the attention mechanism to guide the fusion process. For the detailed network structure, please refer to `model/resnet.py`.*


![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Encoder%20and%20Decoder.jpg)


*Fig .2: The Structure of Encoder and Decoder. For the detailed network structure, please refer to `model/resnet.py`.*

![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Expert%20Structures.jpg)

*Fig. 3: Expert Structures. Detailed implementations of DehazeNet and LowNet can be found in `model/deal/`, while the remaining experts are defined in `model/resnet.py`.*


## 🔹 Dataset
To begin, please first acquire the datasets. This project uses four publicly available infrared-visible image fusion datasets:
- **LLVIP**：http://bupt-ai-cz.github.io/LLVIP/
- **M3FD**：https://github.com/dlut-dimt/TarDAL
- **MSRS**： https://github.com/Linfeng-Tang/MSRS
- **RoadScene**：https://github.com/hanna-xu/RoadScene
  
Please refer to the official sources of each dataset for download and usage instructions.

## 🔹 Model Training

The model’s loss relationships are illustrated in Figure 4 below.


![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Loss.jpg)



*Fig. 4: Loss Function. For implementation details, see `scripts/losses.py` and the loss calls in the training scripts.*


Model training consists of three steps:

1. **Train CLIP with LoRA**  
   Run `lora_clip.py` on visible images to obtain degradation-type embeddings via the CLIP classifier. Save the resulting weights as **W1**.

2. **Joint Segmentation & Fusion Training**  
   Initialize from **W1** and run `train_funtune.py` to optimize both the segmentation and fusion losses. Save the updated weights as **W2**.

3. **Fusion-Only Fine-Tuning**  
   Load both **W1** and **W2**, then run `train.py` to fine-tune exclusively on the fusion loss for final performance gains.

## 🔹 Model Testing

Once training is complete, run `test.py`. Select your target dataset and specify the corresponding weight file paths. After execution, the fused output images will be generated and saved in the `test/` folder.


## 🔹 Result

Partial fusion results of SAGFusion are displayed below, and the fusion outputs for the first five visible–infrared image pairs from each dataset are saved in the **result** folder.


<p align="center">
  <img src="https://github.com/Wohaizainuli/SGAFusion/blob/main/images/result.jpg" alt="fusion result" width="60%" />
  <br/>
  <em>Fig. 5: Fusion Results of SAGFusion</em>
</p>

## 🔹 Contributing & Contact

Thank you for reading and supporting SGAFusion! If you have any questions or encounter issues, feel free to open an issue in this repository or contact me directly at **junjiema_xmtra@163.com**.  
We welcome contributions—please fork the repo, make your changes, and submit a pull request.

