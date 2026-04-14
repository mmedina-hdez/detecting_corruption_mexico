This is the repository to replicate the results of the paper "Learning from sanctioned government suppliers: A machine learning and network science approach to detecting fraud and corruption in Mexico".

The `scripts` folder contains all the python scripts to construct the dataset used for the analysis (`scripts/dataset_creation`) and the ones to replicate the experiments and figures in the main text (`scripts/main_text`).

The figures in the paper can be located in the following jupyter notebooks:

Figure 1. Distribution of contracts and labels across years --> `scripts/main_text/distribution_sanctions.ipynb`

Figure 2. Model's performance (robust metrics) --> `scripts/main_text/evaluation.ipynb`

Figure 3. Top 30 most important features of the model --> `scripts/main_text/shap_interpretation.ipynb`

Figure 4. Mean SHAP values of top 30 across transductive and inductive (EPN and AMLO administration) learning --> `scripts/main_text/shap_interpretation.ipynb`

Figure 5. Selected network features' SHAP dependence plots --> `scripts/main_text/shap_interpretation.ipynb`

Figure 6. Selected Domain-Knowledge features' SHAP dependence plots --> `scripts/main_text/shap_interpretation.ipynb`
