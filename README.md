<h3 align="center">基於自監督對比學習的紡織布料瑕疵研究</h3>

---

<p align="center"> 基於自監督對比學習的紡織布料瑕疵研究
    <br> 
</p>

## 📝 目錄

- [關於](#about)
- [入門](#getting_started)
- [作者](#authors)
- [致謝](#acknowledgement)

## 🧐 關於 <a name = "about"></a>

項目 "基於自監督對比學習的紡織布料瑕疵研究" 是我的畢業項目。

代碼主要是使用 Python 和 PyTorch 框架實現的。該項目的目標是使用自監督對比學習開發一個高效準確的紡織布料瑕疵檢測系統。

該項目僅提供用於設計和訓練SSCL基檢測模型以及評估模型性能的模型和框架。請自行準備您自己的數據集。

## 🏁 入門 <a name = "getting_started"></a>

### 安裝
以下是運行和實驗您的圖像數據集所需遵循的步驟：

1. 準備您的圖像數據集以供使用。
2. 在您的計算機上安裝 'conda'。
3. 打開終端並運行命令 ```conda env create -f environment.yml```。
4. 如果您想要查看注意力圖，請按照以下步驟操作：
   - 在終端上運行命令 ```pip show vision_transformer_pytorch```。
   - 'cd' 到該包的位置。
   - 將 'model.py' 文件替換為 'model/model.py'。
5. 要記錄實驗數據，您可以使用 'w&b' 服務。首先運行命令 ```wandb login``` 登錄。
6. 創建一個項目和相應的sweep。然後將sweep id保存到名為 "sweep_id" 的文件中，放在根目錄中。如果不這樣做，系統將自動創建一個。

### 運行
運行模型訓練的主要檔案是 'tuning.py'
要在後台自動運行實驗，請在終端上運行命令 ```nohup python tuning.py &```。要查看結果，使用終端上的命令 ```tail -f nohup.out```，或在 wandb sweep 頁面上查看結果。

## ✍️ 作者 <a name = "authors"></a>

- 陳建彰 教授 - 淡江大學
- [@Chris Zhan](https://github.com/mikejhan4455) - 項目研究生

## 🎉 致謝 <a name = "acknowledgement"></a>

- 參考文獻
  - [1] 王靖壹. 以深度學習為基礎之紡織布料瑕疵偵測. 淡江大學資訊工程學系碩士論文., 2021.
  - [2] 魏嘉弘. 基於神經網路紡織布料瑕疵快速偵測. 淡江大學資訊工程學系碩士論文., 2022.
  - [3] Guanghua Hu, Junfeng Huang, Qinghui Wang, Jingrong Li, Zhijia Xu, and Xingbiao Huang. Unsupervised fabric defect detection based on a deep convolutional generative adversarial network. Textile Research Journal, 90(3-4):247–270, February 2020.
  - [4] Shuang Mei, Yudan Wang, and Guojun Wen. Automatic Fabric Defect Detection with a Multi-Scale Convolutional Denoising Autoencoder Network Model.
  Sensors, 18(4):1064, April 2018.
  - [5] Jun-Feng Jing, Hao Ma, and Huan-Huan Zhang. Automatic fabric defect detection using a deep convolutional neural network. Coloration Technology, 135(3):213–223, 2019.
  - [6] Juhua Liu, Chaoyue Wang, Hai Su, Bo Du, and Dacheng Tao. Multistage GAN for Fabric Defect Detection. IEEE Transactions on Image Processing, 29:3388–3400, 2020.
  - [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition, December 2015.
  - [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg 37 Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, June 2021.
  - [9] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised Visual Representation Learning by Context Prediction. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 1422–1430, Santiago, Chile, December 2015. IEEE.
  - [10] Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context Encoders: Feature Learning by Inpainting. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2536–2544, June 2016.
  - [11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings of the 37th International Conference on Machine Learning, pages 1597–1607. PMLR, November 2020.
  - [12] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. pages 9726–9735, June 2020.
  - [13] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, koray kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning. In Advances in Neural Information Processing Systems, volume 33, pages 21271–21284. Curran Associates, Inc., 2020. 38
  - [14] Chao Li, Jun Li, Yafei Li, Lingmin He, Xiaokang Fu, and Jingjing Chen. Fabric Defect Detection in Textile Manufacturing: A Survey of the State of the Art. Security and Communication Networks, 2021:e9948808, May 2021.
  - [15] Wei Liu, Xianming Tu, Zhenyuan Jia, Wenqiang Wang, Xin Ma, and Xiaodan Bi. An improved surface roughness measurement method for micro-heterogeneous texture in deep hole based on gray-level co-occurrence matrix and support vector machine. The International Journal of Advanced Manufacturing Technology, 69(1):583–593, October 2013.
  - [16] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84–90, May 2017.
  - [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residua Learning for Image Recognition. In 2016 IEEE Conference on Computer Visio and Pattern Recognition (CVPR), pages 770–778, Las Vegas, NV, USA, June 2016 IEEE.
  - [18] Gregory R. Koch. Siamese Neural Networks for One-Shot Image Recognition. 2015.
  - [19] Lovedeep Gondara. Medical image denoising using convolutional denoising autoencoders. In 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW), pages 241–246, December 2016.
  - [20] Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning. In 2021 IEEE/CVF Conference on Computer Vision and 39 Pattern Recognition (CVPR), pages 16679–16688, Nashville, TN, USA, June 2021. IEEE.
  - [21] Tal Reiss, Niv Cohen, Liron Bergman, and Yedid Hoshen. PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2805–2813. IEEE Computer Society, June 2021.

