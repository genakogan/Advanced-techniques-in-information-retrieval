# <div align="center"><center>Advanced techniques in information retrieval/div>


### Requirements
  1. fitz<br />
  2. pandas<br />
  3. nltk<br />
  4. summarizer<br />
  5. rouge-score<br />
  If you are managing Python packages (libraries) with pip, you can use the configuration file req.txt to install the specified packages with the specified version.<br />

  ### How to Run Your Program (Windows)
1. Open the Cmd, type the following command and press the enter button as shown in the below:<br />
```pip install virtualenv```<br />
2. Create a new virtual environment inside that project's directory to avoid unnecessary issues:<br />
```py -m venv env```<br />
  For more details about virtual environment, please follow the link below:<br />
  <a href="https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/" target="_blank">Virtual Environment</a><br />
3. To use this newly created virtual environment, we just need to activate it. To activate this isolated environment, type the following given command and press enter button as shown below:<br />
```.\env\Scripts\activate```<br />
4. A requirement.txt files include all types of the standard packages and libraries that are used in that particular project. To install all requirements type the following command and press the enter:<br />
```pip install -r HW2022\HW1\req1.txt```<br />
5. The program will be executed from the command line by next command:<br />
```python HW2022\HW1\knn_classifier.py .\HW2022\hhd_dataset```<br />
6. For leaving the virtual environment type the following given command and press enter button as shown below:
  ```deactivate```