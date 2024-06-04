# Control-GIC
This is the Pytorch implementation of the paper "[**Once-for-All: Controllable Generative Image Compression with Dynamic Granularity Adaption**](https://arxiv.org/abs/2406.00758)". We design a unified generative compression model (Control-GIC) capable of variable bitrate adaption across a broad spectrum while preserving high-perceptual fidelity reconstruction. Control-GIC allows one model for variable bitrates and once compression on an entire dataset for constrained bitrate conditions.

![Example](./figs/fig1_00.png)


## 🔓 Installation
```
git clone https://github.com/lianqi1008/Control-GIC.git
cd Control-GIC
conda create -n CGIC python=3.10.4
conda activate CGIC
pip install -r requirements.txt
```
## 🚀 Usage
The **model weights** can be downloaded from [GoogleDrive](https://drive.google.com/file/d/11jaor89-ti6rS2lK2gebg52AEcShFPJk/view?usp=drive_link) and [BaiduNetDisk](https://pan.baidu.com/s/1b_s_UTVQxQRMausjmeCdIg?pwd=byzo).

**Train**
```
python main.py --config configs/config_train.yaml
```
**Inference**

If you want a wide range of bpp (0.1 ~ 0.75), we recommend you start with the Granularity Ratio in [(0, 0.5, 0.5), (0, 0.8, 0.2), (0.1, 0.8, 0.1), (0.3, 0.6, 0.1), (0.5, 0.4, 0.1), (0.9, 0.1, 0)], where each element represents fine-grained, medium-grained, and coarse-grained ratio from left to right, and fine-tune the ratio values based on these ratios to get a finer bpp.

```
python inference.py -i input_dir -o output_dir -w
```
Note that when tuning the ratio to change the bpp, the fine-grained priority is higher than the medium-grained priority, which is higher than the coarse-grained priority. This ensures the best reconstruction quality.
## 💡 Special Capabilities
**Fine control of bitrate**

By fine-tuning the granularity ratio, the model is able to make fine adjustments to the bitrate (Evaluated on the Kodak dataset).

| Granularity Ratio   | Bpp     | LPIPS   |
|:--------------------|:--------|:--------|
| (0.301, 0.599, 0.1) | 0.38925 | 0.03027 | 
| (0.302, 0.598, 0.1) | 0.38983 | 0.03025 |
| (0.303, 0.597, 0.1) | 0.39050 | 0.03020 |
| (0.304, 0.596, 0.1) | 0.39108 | 0.03014 |

**Constant bitrate**

Once the granularity ratio is fixed, the bpp values of each image in the dataset are very close to each other, thus easily fulfilling the constrained bandwidth and bitrate conditions. For example, in one compression on the Kodak dataset, we set the ratio to (0.1, 0.8, 0.1), and obtained 24 images with bpp ranging from **0.262** to **0.277**, with only slight variations. 

## Cite


## Thanks
The code references [VQGAN](https://github.com/CompVis/taming-transformers), [MoVQGAN](https://github.com/ai-forever/MoVQGAN), [DQ-VAE](https://github.com/CrossmodalGroup/DynamicVectorQuantization) and Huffman coding from Bhrigu Srivastava@bhrigu123. Thanks for these excellent works!
