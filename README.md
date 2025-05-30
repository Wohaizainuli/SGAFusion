# SGAFusion：Semantic-Guided Adaptive Fusion for Infrared-Visible Images Under Degradation Conditions

> This repository provides a non-official and simplified version of our implementation, intended solely for academic communication and preliminary understanding.  
> To protect the originality of our work and ensure compliance with data and intellectual property considerations, the **complete codebase will be released after the paper is officially published**.
> We appreciate your understanding and support.

## 🔹 Framework Overview

The structure of **SGAFusion** is illustrated in the figure below:


![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Algorithm%20Framework.jpg)


*Fig .1: Overall architecture of SGAFusion. Semantic information representing degradation types is automatically generated from the visible image via the CLIP model, and used as the query (Q) in the attention mechanism to guide the fusion process.*


![image](https://github.com/Wohaizainuli/SGAFusion/blob/main/images/Algorithm%20Framework.jpg)


*Fig .2: The structure of Encoder and Decoder
## 🔹 Dataset
To begin, please first acquire the datasets. This project uses four publicly available infrared-visible image fusion datasets:
- **LLVIP**：http://bupt-ai-cz.github.io/LLVIP/
- **M3FD**：https://github.com/dlut-dimt/TarDAL
- **MSRS**： https://github.com/Linfeng-Tang/MSRS
- **RoadScene**：https://github.com/hanna-xu/RoadScene
  
Please refer to the official sources of each dataset for download and usage instructions.

