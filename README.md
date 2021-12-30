# Startup Classification-Recommender System: Project Overview

* Created a recommender app that erecommends startups to help investors know where they can put their money on.
* Processed and transformed Crunchbase data
* Engineered features from objects(companies and products data).
* Optimized MLP, LightGBM, XGBoost and Random Forest Classifiers using GridsearchCV to reach the best model. 
* Implemented a hybrid recommender system based on KNN algorithm.
* Built a client facing API using Streamlit framework

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, streamlit, joblib  
**For Project Framework Requirements:**  ```pip install -r requirements.txt```  
**Data on Kaggle:** https://www.kaggle.com/justinas/startup-investments   
**Streamlit Productionization:** https://towardsdatascience.com/deploying-a-basic-streamlit-app-ceadae286fd0
**Heroku Deployment:** https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd

## Data Cleaning and Processing
After collecting the data, I cleaned it up so that it was usable for our model. I made the following changes and created the following variables:

*	Age from date of founding 
*	Ageset from classifying the feature age
*	Continent from the country_code 
*	Number of products(kind of hot encoded)
*	Number of funding rounds(kind of hot encoded) 
*	Category from category_code reclassification

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 30%.   

I tried four different models and evaluated them using accuracy score. I chose accuracy score because it is relatively easy to interpret based on the correct classifying.   

I tried four different models:
*	**Multi-layer Perceptron Classifier** – Experiment of a neural network based on Sci-kit Learn
*	**XGBoost** – Because of the categorical variables, I thought a boosted model will be efficient
*	**Random Forest** – Again, with the categorical variables and the efficiency it offers on a wide range of data arrangement, I thought that this would be a good fit.
*   **LightGBM** - Because of its accuracy and speeds

## Model performance

The Random Forest model far outperformed the other approaches on the test and validation sets.
*   **MLP Classifier**: Score = 0.8911
*	**RF Classifier** :  Score = 0.8927
*	**XGBoost Classifier**: Score = 0.8931
*	**LightGBM Classifier**: Score = 0.8933

## Recommender Assembling

In this step, I built a KNN based recommender that borrows ideas from content and collaborative filtering.
## Productionization

In this step, I built a Streamlit API endpoint that was hosted on heroku by following along with the productization tutorial in the reference section above together with the official [Streamlit documentation](https://docs.streamlit.io/en/stable/). The API endpoint takes in a request with status, chance of success and number of startups values from a user and returns recommended startups and their responding shallow profile.