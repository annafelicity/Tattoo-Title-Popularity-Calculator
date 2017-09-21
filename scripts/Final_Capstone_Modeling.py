
# coding: utf-8

# In[1]:

import pandas as pd
import re
from textblob import TextBlob
from textstat.textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
get_ipython().magic('matplotlib inline')


# In[2]:

data_1500_kf = pd.read_csv("full_data/english_books_final_deduped_2_to_1500.csv", usecols=["age_in_2017", "genre", "is_non_adult", "proper_title", "extracted_libcount"])


# In[3]:

data_1500_kf.head()


# In[4]:

data_1500_kf.shape


# ### add the title-based features to the 2-1500 set

# In[5]:

data_1500_kf.isnull().sum()


# In[6]:

data_1500_kf.dropna(inplace=True)


# In[7]:

data_1500_kf.index = range(len(data_1500_kf))


# In[8]:

data_1500_kf.shape


# In[9]:

#add sentiment analysis
def text_blob_sentiment_polarity(value):
    blob = TextBlob(value)
    return blob.sentiment.polarity
def text_blob_sentiment_subjectivity(value):
    blob = TextBlob(value)
    return blob.sentiment.subjectivity


# In[10]:

data_1500_kf["sentiment_polarity"] = data_1500_kf["proper_title"].apply(text_blob_sentiment_polarity)


# In[11]:

data_1500_kf["sentiment_subjectivity"] = data_1500_kf["proper_title"].apply(text_blob_sentiment_subjectivity)


# In[12]:

#add reading level
def reading_level_comp(string):
    try:
        level = textstat.text_standard(string)
        return level
    except:
        return "Unclear"


# In[13]:

data_1500_kf["reading_level"] = data_1500_kf["proper_title"].apply(reading_level_comp)


# In[14]:

#make reading level into a dummies df
lb_rl = LabelBinarizer()
reading_level_lb = lb_rl.fit_transform(data_1500_kf["reading_level"])


# In[15]:

reading_level_dummies = pd.DataFrame(reading_level_lb)


# In[16]:

reading_level_dummies.shape


# In[17]:

#add number of words column
data_1500_kf["number_of_words"] = data_1500_kf["proper_title"].apply(lambda x: len(x.split()))


# In[18]:

#add title length
data_1500_kf["title_length"] = data_1500_kf["proper_title"].apply(lambda x: len(x))


# In[19]:

#this is not a title feature but may use
def fiction_binarizer(value):
    if "Fiction" in value:
        return 1
    elif "Poetry" in value:
        return 1
    elif "Drama" in value:
        return 1
    else:
        return 0


# In[20]:

data_1500_kf["is_fiction"] = data_1500_kf["genre"].apply(fiction_binarizer)


# In[21]:

#add topic modeling
cv_for_lda = CountVectorizer(min_df=5, max_df=.75, ngram_range=(1,3), stop_words="english")

words = cv_for_lda.fit_transform(data_1500_kf["proper_title"])


# In[22]:

lda_8 = LatentDirichletAllocation(n_topics=8, max_iter=15,
                                topic_word_prior=2,
                                learning_offset=50., random_state=3)

lda_8.fit(words)


# In[23]:

transformed_data_8= lda_8.transform(words)
transformed_data_8 = pd.DataFrame(transformed_data_8, columns=['Topic %s' % x for x in range(8)])


# In[24]:

def top_topic_number_extractor(dataframe):
    top_topic_list = []
    for i in dataframe.index:
        ordered_row = dataframe.iloc[i,:].sort_values(ascending=False)
        top_topic_name = ordered_row.index[0]
        count_pattern = re.compile("\d+")
        top_topic_number = count_pattern.search(top_topic_name).group()
        top_topic_list.append(int(top_topic_number))
    return top_topic_list


# In[25]:

data_1500_kf["top_topic_number_lda8"] = top_topic_number_extractor(transformed_data_8)


# In[26]:

#make dummy variable columns for top topic number
lb = LabelBinarizer()
topic_lb = lb.fit_transform(data_1500_kf["top_topic_number_lda8"])


# In[27]:

#make df with categoricalized top topics
top_topics_df = pd.DataFrame(topic_lb)
top_topics_df.head()


# In[28]:

#tfidf vectorize the words:
#all lowercase consistently worked worse, so keeping capitalization here
#giving this a test now with stopwords--still unclear if they should stay
tfidf = TfidfVectorizer(min_df=5, max_df=.95, lowercase=False, stop_words="english", ngram_range=(1,3))
tfidf_title = tfidf.fit_transform(data_1500_kf["proper_title"])


# In[29]:

len(tfidf.vocabulary_)


# In[30]:

#make it into a df to join onto beginning data:
tfidf_title_df = pd.DataFrame(tfidf_title.todense(), 
                  columns=tfidf.get_feature_names())


# In[31]:

tfidf_title_df.shape


# In[32]:

data_1500_kf.head()


# In[33]:

data_1500_kf.shape


# ### create y and X for modeling

# In[34]:

y = data_1500_kf["extracted_libcount"]


# In[35]:

X = pd.concat([data_1500_kf[["age_in_2017", "is_non_adult", "sentiment_polarity", "sentiment_subjectivity", "number_of_words", "title_length", "is_fiction"]], top_topics_df, tfidf_title_df, reading_level_dummies], axis=1)


# In[52]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)


# In[37]:

rfr = RandomForestRegressor(min_samples_leaf=5, n_estimators=20)
rfr.fit(X_train, y_train)


# In[38]:

rfr.score(X_train, y_train)


# In[39]:

rfr.score(X_test, y_test)


# In[40]:

predicted_vals = rfr.predict(X_test)


# In[41]:

explained_variance_score(predicted_vals, y_test)


# In[42]:

mean_absolute_error(predicted_vals, y_test)


# In[43]:

mean_squared_error(predicted_vals, y_test)


# In[44]:

plt.scatter(y_test, predicted_vals)


# ### try with different features

# In[36]:

#just age and title stuff:
X2 = pd.concat([data_1500_kf[["age_in_2017", "sentiment_polarity", "sentiment_subjectivity", "number_of_words", "title_length"]], top_topics_df, tfidf_title_df, reading_level_dummies], axis=1)


# In[37]:

X2.shape


# In[38]:

X2.head()


# In[39]:

X2_train, X2_test = train_test_split(X2, test_size=0.33, random_state=3)


# In[49]:

rfr_2 = RandomForestRegressor(min_samples_leaf=5, n_estimators=20)
rfr_2.fit(X2_train, y_train)


# In[50]:

rfr_2.score(X2_train, y_train)


# In[51]:

rfr_2.score(X2_test, y_test)


# In[52]:

predicted_vals_2 = rfr_2.predict(X2_test)


# In[53]:

explained_variance_score(predicted_vals_2, y_test)


# In[54]:

mean_absolute_error(predicted_vals_2, y_test)


# In[55]:

mean_squared_error(predicted_vals_2, y_test)


# In[56]:

plt.scatter(y_test, predicted_vals_2)
plt.ylim(0,1400)


# ### try without age

# In[57]:

#just title stuff:
X3 = pd.concat([data_1500_kf[["sentiment_polarity", "sentiment_subjectivity", "number_of_words", "title_length"]], top_topics_df, tfidf_title_df, reading_level_dummies], axis=1)


# In[58]:

X3_train, X3_test = train_test_split(X3, test_size=0.33, random_state=3)


# In[59]:

rfr_3 = RandomForestRegressor(min_samples_leaf=5, n_estimators=20)
rfr_3.fit(X3_train, y_train)


# In[60]:

rfr_3.score(X3_train, y_train)


# In[61]:

rfr_3.score(X3_test, y_test)


# In[62]:

# ### GridSearch on data set 1
# parameters = {"n_estimators": [10, 20, 50, 100, 200], "max_features": [.30, .50, .70, .90], "max_depth": [5, 10, 15, 20], "min_samples_leaf": [2, 3, 4, 5, 7, 10]}
# rf = RandomForestRegressor()
# grid = GridSearchCV(rf, parameters)


# In[63]:

#commenting this out for now, since it takes so long to run
# grid.fit(X_train, y_train)


# In[64]:

#best score was: 0.054772596880442219
# grid.best_score_


# In[65]:

#best params were: 
# {'max_depth': 15,
#  'max_features': 0.3,
#  'min_samples_leaf': 10,
#  'n_estimators': 100}
#see below for model run with best params
# grid.best_params_


# In[66]:

# rfr_best_params = RandomForestRegressor(max_depth=15, max_features=0.3, min_samples_leaf=10, n_estimators=100)
# rfr_best_params.fit(X_train, y_train)


# In[67]:

# rfr_best_params.score(X_train, y_train)


# In[68]:

# rfr_best_params.score(X_test, y_test)


# In[69]:

# rfr_bp_predictions = rfr_best_params.predict(X_test)


# In[70]:

# mean_absolute_error(y_test, rfr_bp_predictions)


# In[71]:

# plt.scatter(y_test, rfr_bp_predictions)


# In[72]:

### GridSearch on data set 2
parameters = {"n_estimators": [10, 20, 50, 100, 200], "max_features": [.10, .30, .50, .90], "max_depth": [5, 10, 15, 20], "min_samples_leaf": [3, 5, 7, 10, 15]}
rf = RandomForestRegressor()
grid_2 = GridSearchCV(rf, parameters)


# In[90]:

grid_2.fit(X2_train, y_train)


# In[91]:

grid_2.best_score_


# In[92]:

grid_2.best_params_


# In[54]:

rfr_best_params_2 = RandomForestRegressor(max_depth=20, max_features=0.3, min_samples_leaf=5, n_estimators=100)
rfr_best_params_2.fit(X2_train, y_train)


# In[55]:

rfr_bp2_predictions = rfr_best_params_2.predict(X2_test)


# In[56]:

rfr_best_params_2.score(X2_test, y_test)


# In[57]:

mean_absolute_error(y_test, rfr_bp2_predictions)


# In[58]:

explained_variance_score(y_test, rfr_bp2_predictions)


# In[59]:

mean_squared_error(y_test, rfr_bp2_predictions)


# In[60]:

plt.scatter(y_test, rfr_bp2_predictions, marker="+", s=1, facecolor='#F557A4')
plt.ylim(1,1500)
plt.xlabel("Actual Total Library Counts")
plt.ylabel("Predicted Total Library Counts")
plt.title("Comparison of Actual vs. Predicted Library Counts", color="#840000")
plt.savefig("images/actual_vs_predicted_libcount_test.png", dpi=150)


# In[61]:

plt.scatter(y_test, rfr_bp2_predictions, marker="+", s=1, facecolor='#F557A4')
plt.ylim(1,400)
plt.xlim(1,400)
plt.xlabel("Actual Total Library Counts")
plt.ylabel("Predicted Total Library Counts")
plt.title("Comparison of Actual vs. Predicted Library Counts", color="#840000")
plt.savefig("images/actual_vs_predicted_libcount_detail_test.png", dpi=150)


# In[100]:

features = X2_train.columns


# In[101]:

feature_imp_df = pd.DataFrame([features, rfr_best_params_2.feature_importances_]).transpose()
feature_imp_df.sort_values(by=1, ascending=False).head(20)


# ### fit the model to the entire dataset

# In[40]:

rfr_final = RandomForestRegressor(max_depth=20, max_features=0.3, min_samples_leaf=5, n_estimators=100)
rfr_final.fit(X2, y)


# In[41]:

features_final = X2.columns


# In[105]:

feature_imp_final = pd.DataFrame([features_final, rfr_final.feature_importances_]).transpose()
feature_imp_final.sort_values(by=1, ascending=False).head(20)


# In[42]:

#add scoring on whole model
predictions = rfr_final.predict(X2)


# In[45]:

rfr_final.score(X2, y)


# In[46]:

explained_variance_score(y, predictions)


# In[47]:

mean_absolute_error(y, predictions)


# In[48]:

mean_squared_error(y, predictions)


# In[49]:

plt.scatter(y, predictions, marker="+", s=1, facecolor='#F557A4')
plt.ylim(1,1500)
plt.xlabel("Actual Total Library Counts")
plt.ylabel("Predicted Total Library Counts")
plt.title("Comparison of Actual vs. Predicted Library Counts", color="#840000")
plt.savefig("images/actual_vs_predicted_libcount.png", dpi=150)


# In[50]:

plt.scatter(y, predictions, marker="+", s=1, facecolor='#F557A4')
plt.ylim(1,400)
plt.xlim(1,400)
plt.xlabel("Actual Total Library Counts")
plt.ylabel("Predicted Total Library Counts")
plt.title("Comparison of Actual vs. Predicted Library Counts", color="#840000")
plt.savefig("images/actual_vs_predicted_libcount_detail.png", dpi=150)


# In[ ]:




# In[ ]:




# In[ ]:




# In[132]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos.csv"))


# In[133]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos_5.csv"))


# In[134]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos_10.csv"))


# In[135]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_I_love_cats.csv"))


# In[136]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock.csv"))


# In[137]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock_2.csv"))


# In[138]:

rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock_10.csv"))


# In[ ]:




# ### try the gradient boosting regressor just for kicks

# In[ ]:

gbr = GradientBoostingRegressor()
gbr.fit(X2_train, y_train)


# In[ ]:

gbr.score(X2_train, y_train)


# In[ ]:

gbr.score(X2_test, y_test)


# In[ ]:

gbr_predictions = gbr.predict(X2_test)


# In[ ]:

explained_variance_score(gbr_predictions, y_test)


# In[ ]:

mean_absolute_error(gbr_predictions, y_test)


# In[ ]:

mean_squared_error(gbr_predictions, y_test)


# In[ ]:

plt.scatter(y_test, gbr_predictions)


# In[ ]:

from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# In[ ]:

# parameters = {"min_samples_leaf": range(1,20), "max_depth": range(1,50), "max_features": np.arange( 0.0, 1.0+0.0, 0.1 ).tolist(), "n_estimators": range(1,500)}
# rscv = RandomizedSearchCV(gbr, parameters, verbose=10)


# In[ ]:

# rscv.fit(X2_train, y_train)


# In[ ]:

# #was 0.016660259457341956
# rscv.best_score_


# In[ ]:

# #were:
# {'max_depth': 5,
#  'max_features': 0.1,
#  'min_samples_leaf': 7,
#  'n_estimators': 96}
# rscv.best_params_


# In[ ]:

#NOTE: As with above need to redo this RandomizedSearch!!!!!
gbr_be = GradientBoostingRegressor(max_depth=5, max_features=0.10, min_samples_leaf=7, n_estimators=100)
gbr_be.fit(X2_train, y_train)


# In[ ]:

gbr_be.score(X2_test, y_test)


# In[ ]:

gbr_be_predictions = gbr_be.predict(X2_test)


# In[ ]:

explained_variance_score(y_test, gbr_be_predictions)


# In[ ]:

mean_absolute_error(y_test, gbr_be_predictions)


# In[ ]:

plt.scatter(y_test, gbr_be_predictions, marker="+", s=1)
plt.ylim(1,1400)
plt.figure(figsize=(20,10), dpi=150)


# ### Note: here I re-run the model with max_depth increased because the above seemed like it was plateau-ing, ideally I'd like to do a GridSearch now using the below as a stepping off point for a narrow range of values than what was in the RandomizedSearch

# In[ ]:

# #re-run with max_depth adjusted:
gbr_2 = GradientBoostingRegressor(max_depth=10, max_features=0.10, min_samples_leaf=10, n_estimators=100)
gbr_2.fit(X2_train, y_train)


# In[ ]:

gbr_2_predictions = gbr_2.predict(X2_test)


# In[ ]:

gbr_2.score(X2_test, y_test)


# In[ ]:

mean_absolute_error(y_test, gbr_2_predictions)


# In[ ]:

explained_variance_score(y_test, gbr_2_predictions)


# In[ ]:

plt.scatter(y_test, gbr_2_predictions, marker="+", s=1)
plt.ylim(0,1400)


# In[ ]:



