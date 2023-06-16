
<h3 align="center">Research of Self-Supervised Contrastive Learning-based Textile Fabric Defect</h3>

---

<p align="center"> 基於神經網路紡織布料瑕疵快速偵測
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

continue writing after in english: The project "Research of Self-Supervised Contrastive Learning-based Textile Fabric Defect"(SSCL-based Textile Fabric Defect Detection) is my graduation project.

The codes are mainly implemented in Python and PyTorch framework. The goal of the project is to develop an efficient and accurate defect detection system for textile fabrics using self-supervised contrastive learning.

The project will provide the model and framework ONLY for designing and training the SSCL-based detection model, and evaluating the model's performance. Please prepare your own dataset. 



## 🏁 Getting Started <a name = "getting_started"></a>



### Installing
Here are the steps to be followed to run and experiment on your image dataset:

1. Prepare your image dataset for use.
2. Install 'conda' on your computer.
3. Open your terminal and run the command ```conda env create -f environment.yml```.
4. If you want to see the Attention maps, follow these steps: 
   - Run the command ```pip show vision_transformer_pytorch``` on your terminal.
   - Navigate to the location of the package.
   - Replace the 'model.py' file with 'model/model.py'.
5. To log experiment data, you can use the 'w&b' service. Login with the command ```wandb login``` first.
6. Create a project and corresponding sweep. Then save the sweep id into a file named "sweep_id" in the root folder. If you don't, the system will create one automatically.


### Running
To run the experiment automatically in the background, run the command ```nohup python tuning.py &``` on your terminal. To see the results, use the command ```tail -f nohup.out``` on your terminal or view them on the wandb sweep page.


## ✍️ Author <a name = "authors"></a>

- [@Chris Zhan](https://github.com/mikejhan4455) - Project Student


## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- References
  - The references will be listed later.
