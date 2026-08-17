\# SnappFood Sentiment Analysis



An intelligent sentiment analysis project for analyzing Persian customer reviews from SnappFood using machine learning, deep learning, and transformer-based language models.



\## Overview



Customer reviews contain valuable information about user satisfaction and can help businesses understand customer opinions and improve their services.



This project focuses on sentiment analysis of Persian SnappFood reviews. Several machine learning and deep learning approaches were implemented and compared, including traditional machine learning models, recurrent neural networks, CNN-based architectures, and the ParsBERT transformer model.



\## Objectives



\* Classify Persian customer reviews based on sentiment

\* Compare traditional machine learning and deep learning approaches

\* Apply Persian-specific text preprocessing techniques

\* Evaluate transformer-based models for Persian sentiment analysis

\* Provide a foundation for monitoring and analyzing customer sentiment trends



\## Dataset



The project uses a Persian sentiment analysis dataset containing approximately 70,000 SnappFood customer reviews.



> The dataset is not included in this repository due to its size.



\## Text Preprocessing



Persian text preprocessing was performed using the \*\*Hazm\*\* natural language processing library.



The preprocessing pipeline includes:



\* Text normalization

\* Tokenization

\* Stemming

\* Lemmatization

\* Text cleaning

\* Preparing text for machine learning and deep learning models



\## Models



\### Traditional Machine Learning



\* Logistic Regression

\* Random Forest

\* XGBoost



\### Deep Learning



\* RNN

\* GRU

\* LSTM

\* Bidirectional LSTM (BiLSTM)

\* TextCNN



\### Transformer



\* ParsBERT



ParsBERT was used to leverage contextual representations specifically designed for the Persian language.



\## Results



The implemented models were evaluated using classification performance metrics.



The Logistic Regression model achieved an accuracy of approximately \*\*82%\*\* on the evaluated dataset.



The project also explores more advanced deep learning and transformer-based architectures, including LSTM, BiLSTM, TextCNN, and ParsBERT.



\## Technologies



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* XGBoost

\* PyTorch

\* Hazm

\* Transformers

\* ParsBERT

\* Jupyter Notebook

\* Streamlit



\## Project Structure



```text

snappfood-sentiment-analysis/

│

├── ParsBert\_train.ipynb

├── snapfood\_sentiment (final).ipynb

├── models.py

├── preprocess.py

├── ui.py

├── requirements.txt

├── README.md

└── .gitignore

```



Large files such as the dataset, trained models, and virtual environment are excluded from the repository using `.gitignore`.



\## Installation



Clone the repository:



```bash

git clone https://github.com/azamkoo/snappfood-sentiment-analysis.git

cd snappfood-sentiment-analysis

```



Create a virtual environment:



```bash

python -m venv venv

```



Activate the virtual environment on Windows:



```bash

venv\\Scripts\\activate

```



Install the dependencies:



```bash

pip install -r requirements.txt

```



\## Usage



The Jupyter notebooks contain the main data preprocessing, model training, and evaluation workflow.



To run the Streamlit interface:



```bash

streamlit run ui.py

```



\## Future Improvements



\* Fine-tune ParsBERT for improved Persian sentiment classification

\* Perform hyperparameter optimization

\* Add more evaluation metrics

\* Address potential class imbalance

\* Expand the dataset

\* Improve the Streamlit dashboard

\* Add real-time sentiment monitoring and visualization

\* Deploy the application as a web service



\## Author



\*\*Azam Kooravand\*\*



Computer Engineering Graduate



GitHub: \[azamkoo](https://github.com/azamkoo)



