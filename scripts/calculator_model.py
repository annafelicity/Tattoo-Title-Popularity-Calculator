
# coding: utf-8

## Note: this is the beginnings of working this out into some sort of model that can be used in a web app where a user would input a title and it would get run through this model

import pandas as pd
import re
from textblob import TextBlob
from textstat.textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer

#I think I can rework this now to just import the two relevant columns (age and proper title for making X and then just import "extracted_libcount" for y)
data_1500_kf = pd.read_csv("full_data/english_books_final_deduped_2_to_1500.csv", usecols=["age_in_2017", "proper_title", "extracted_libcount"])

#drop null rows (there are only a few) and reindex

data_1500_kf.dropna(inplace=True)
data_1500_kf.index = range(len(data_1500_kf))


## add the title-based features to the 2-1500 set

#add sentiment analysis
def text_blob_sentiment_polarity(value):
    blob = TextBlob(value)
    return blob.sentiment.polarity
def text_blob_sentiment_subjectivity(value):
    blob = TextBlob(value)
    return blob.sentiment.subjectivity

data_1500_kf["sentiment_polarity"] = data_1500_kf["proper_title"].apply(text_blob_sentiment_polarity)

data_1500_kf["sentiment_subjectivity"] = data_1500_kf["proper_title"].apply(text_blob_sentiment_subjectivity)

#add reading level
def reading_level_comp(string):
    try:
        level = textstat.text_standard(string)
        return level
    except:
        return "Unclear"

data_1500_kf["reading_level"] = data_1500_kf["proper_title"].apply(reading_level_comp)


#make reading level into a dummies df
lb_rl = LabelBinarizer()
reading_level_lb = lb_rl.fit_transform(data_1500_kf["reading_level"])

reading_level_dummies = pd.DataFrame(reading_level_lb)


#add number of words column
data_1500_kf["number_of_words"] = data_1500_kf["proper_title"].apply(lambda x: len(x.split()))


#add title length
data_1500_kf["title_length"] = data_1500_kf["proper_title"].apply(lambda x: len(x))


#add topic modeling
cv_for_lda = CountVectorizer(min_df=5, max_df=.75, ngram_range=(1,3), stop_words="english")

words = cv_for_lda.fit_transform(data_1500_kf["proper_title"])


lda_8 = LatentDirichletAllocation(n_topics=8, max_iter=15,
                                topic_word_prior=2,
                                learning_offset=50., random_state=3)

lda_8.fit(words)


transformed_data_8= lda_8.transform(words)
transformed_data_8 = pd.DataFrame(transformed_data_8, columns=['Topic %s' % x for x in range(8)])

#extract top topic number for each title
def top_topic_number_extractor(dataframe):
    top_topic_list = []
    for i in dataframe.index:
        ordered_row = dataframe.iloc[i,:].sort_values(ascending=False)
        top_topic_name = ordered_row.index[0]
        count_pattern = re.compile("\d+")
        top_topic_number = count_pattern.search(top_topic_name).group()
        top_topic_list.append(int(top_topic_number))
    return top_topic_list


data_1500_kf["top_topic_number_lda8"] = top_topic_number_extractor(transformed_data_8)


#make dummy variables for top topic number
lb = LabelBinarizer()
topic_lb = lb.fit_transform(data_1500_kf["top_topic_number_lda8"])


#make df with categoricalized top topics
top_topics_df = pd.DataFrame(topic_lb)
top_topics_df.head()


#tfidf vectorize the words:
#all lowercase consistently worked worse, so keeping capitalization here
#current model uses stopwords, although a custom stopword list is something to experiment with in the future
tfidf = TfidfVectorizer(min_df=5, max_df=.95, lowercase=False, stop_words="english", ngram_range=(1,3))
tfidf_title = tfidf.fit_transform(data_1500_kf["proper_title"])


#make tfidf into a df to join onto beginning data:
tfidf_title_df = pd.DataFrame(tfidf_title.todense(), 
                  columns=tfidf.get_feature_names())


# ### create y and X for modeling, note: called X2 here because it was the second version that ended up working out the best

y = data_1500_kf["extracted_libcount"]


#X2 had just age and all the title features, not genre features which didn't seem to matter:
X2 = pd.concat([data_1500_kf[["age_in_2017", "sentiment_polarity", "sentiment_subjectivity", "number_of_words", "title_length"]], top_topics_df, tfidf_title_df, reading_level_dummies], axis=1)


## After many experiments as to what parameters were best, I fit the model to the entire dataset

rfr_final = RandomForestRegressor(max_depth=20, max_features=0.3, min_samples_leaf=5, n_estimators=100)
rfr_final.fit(X2, y)

#And then this was my janky MVP way of getting it to predict the library count using the one-line csvs from the proto-pipeline
rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos_5.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_Ancient_Inkc_the_historical_mystery_of_tattoos_10.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_I_love_cats.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock_2.csv"))


rfr_final.predict(pd.read_csv("test_data/test titles/X2_Tattoos_Rock_10.csv"))