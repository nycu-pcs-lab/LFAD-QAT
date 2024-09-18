## LFADS Quantization-Aware Training (QAT)

This repository extends the **Targeted Neural Dynamical Modeling** project by incorporating **Quantization-Aware Training (QAT)** using **QKeras**. The goal of this project is to enable efficient deployment on **FPGA** platforms.

### Key Features:
- Built on top of the **LFADS (Latent Factor Analysis via Dynamical Systems)** framework.
- Extended for **Quantization-Aware Training** to improve hardware efficiency without sacrificing model performance.
- Designed for deployment on FPGAs, with plans to convert the trained model into FPGA firmware using **hls4ml**.

### Notebooks:
- For training LFADS with higher bit widths (16, 14, 12, 10 bits), run the notebook:  
  `notebook/lfads_qat_higher_bits.ipynb`
- For lower bit widths (8, 6, 4 bits), run the notebook:  
  `notebook/lfads_qat_lower_bits.ipynb`

After running these notebooks, you'll obtain a **trained QKeras model**.

To proceed with converting the trained model into an **hls4ml** project for FPGA deployment, use the notebook:  
`notebook/convert_qlfads_hls4ml.ipynb`

### Requirements:
- **Python 3.11**
- **TensorFlow 2.8.0**

### Future Work:
- Conversion of the trained model into FPGA firmware using **hls4ml**.
- The project environment file (`environment.yml`) for **conda** is in preparation and will be provided soon.

### Notes:
- The code still requires some organization, and improvements are ongoing.
