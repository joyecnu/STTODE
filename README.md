# STTODE: Spatial-Temporal Transformer Ordinary Differential Equation Networks for Pedestrian Trajectory Forecasting
Official PyTorch code for ICME 2025 paper "STTODE: Spatial-Temporal Transformer Ordinary Differential Equation Networks for Pedestrian Trajectory Forecasting".


Abstract—The complexity of Spatial-Temporal relationships and the ambiguous geometric structure in pedestrian interactions pose significant challenges for pedestrian trajectory forecasting. Existing Transformer-based methods for pedestrian trajectory prediction rely on the multi-head self-attention mechanism, which measures similarity based solely on the Euclidean distance of embedding vectors. However, these methods overlook the non-Euclidean geometric relationships inherent in pedestrian interactions, which are crucial for accurate trajectory forecasting. To address this challenge, we propose Spatial-Temporal Transformer Ordinary Differential Equation Networks (STTODE). Specifically, to better capture the Spatial-Temporal dependencies in neural networks, we introduce a Transformer based on Multi-Head Geodesic Self-Attention (MHGSA) to model non-Euclidean interactions in pedestrian trajectories. Additionally, we apply tensor-based Ordinary Differential Equations (ODEs) to capture the Spatial-Temporal dynamics, enabling deeper network construction while simultaneously leveraging Spatial-Temporal features. Experimental results on several widely used pedestrian trajectory pubulic datasets, including ETH-UCY, SDD and NBA, demonstrate that our approach surpasses outstanding baselines. 

#录用决定
![image](https://github.com/user-attachments/assets/223172c1-17e6-48bd-884a-0e27e67ace4d)
![image](https://github.com/user-attachments/assets/be24bc06-12e9-4b11-8eeb-eddcc19418ca)
more details in the pdf.
# SHOW


https://github.com/user-attachments/assets/2985e28b-1dd2-4bd9-88a0-8801161731fd


# Requirement
Recommend Environment
Tested OS: Linux / RTX 3090
Python == 3.7.11
PyTorch == 1.8.1+cu111

# Dependencies
install the dependencies from the requirements.txt:



# model
![image](https://github.com/user-attachments/assets/eb4b01fe-7e44-4917-9345-54e101496387){width="200"}
![image](https://github.com/user-attachments/assets/75776bda-c7e0-4e38-ab0c-235d283d8d47)


