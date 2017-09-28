
# coding: utf-8

# Note: this is the prototype of the various bits of code that would need to be in a pipeline for user inputs
# 
# I need to do a lot more work on this, pickling the sub-models, etc. Stay tuned!

# In[1]:

import pandas as pd
import re
from titlecase import titlecase
from textblob import TextBlob
from textstat.textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelBinarizer


#read in the csv for modeling and drop null rows and reindex
data_1500_kf = pd.read_csv("full_data/english_books_final_deduped_2_to_1500.csv", usecols=["proper_title", "age_in_2017"])
data_1500_kf.dropna(inplace=True)
data_1500_kf.index = range(len(data_1500_kf))

#sample user input--age will need to be automatically input as 0 as user will just enter title
x = [("Marking Identity: Maori Tattoos and Cultural History", 0)]

#turn user input into df
df_x = pd.DataFrame(x, columns=["title", "age_in_2017"])

#turn title into proper title
def make_proper_title(string):
    string = string.replace(" : ", ": ")
    string = string.rstrip(".")
    return titlecase(string)

df_x["proper_title"] = df_x["title"].apply(make_proper_title)

#add sentiment analysis
def text_blob_sentiment_polarity(value):
    blob = TextBlob(value)
    return blob.sentiment.polarity
def text_blob_sentiment_subjectivity(value):
    blob = TextBlob(value)
    return blob.sentiment.subjectivity

df_x["sentiment_polarity"] = df_x["proper_title"].apply(text_blob_sentiment_polarity)

df_x["sentiment_subjectivity"] = df_x["proper_title"].apply(text_blob_sentiment_subjectivity)

#add reading level
def reading_level_comp(string):
    try:
        level = textstat.text_standard(string)
        return level
    except:
        return "Unclear"

df_x["reading_level"] = df_x["proper_title"].apply(reading_level_comp)


#need to make a dataframe for dummies based on the training set
#first get the training reading levels to use for dummies
#this is NOT the way to do this--takes too long
#somehow I need to pickle this or something
data_1500_kf["reading_level"] = data_1500_kf["proper_title"].apply(reading_level_comp)

#make reading level from training set into a dummies df
lb_rl = LabelBinarizer()
lb_rl.fit_transform(data_1500_kf["reading_level"])

#use the lb for the input title
lb_rl_input = lb_rl.transform(df_x["reading_level"])
lb_rl_input

reading_level_dummies = pd.DataFrame(lb_rl_input)


#add number of words column
df_x["number_of_words"] = df_x["proper_title"].apply(lambda x: len(x.split()))


#add title length
df_x["title_length"] = df_x["proper_title"].apply(lambda x: len(x))


#add topic modeling
#make the CV model on the training set
#NOTE: I can pickle this model for use in both the input pipeline and the model
cv_for_lda = CountVectorizer(min_df=5, max_df=.75, ngram_range=(1,3), stop_words="english")

words = cv_for_lda.fit_transform(data_1500_kf["proper_title"])


#do the topic modeling on the training set
#NOTE: as above re: pickling
lda_8 = LatentDirichletAllocation(n_topics=8, max_iter=15,
                                topic_word_prior=2,
                                learning_offset=50., random_state=3)

lda_8.fit(words)


#CountVectorize the words in the input string in keeping with the topic modeling model
input_words = cv_for_lda.transform(df_x["proper_title"])


#transform the input string using the training set model
transformed_data_8= lda_8.transform(input_words)
transformed_data_8 = pd.DataFrame(transformed_data_8, columns=['Topic %s' % x for x in range(8)])


def top_topic_number_extractor(dataframe):
    top_topic_list = []
    for i in dataframe.index:
        ordered_row = dataframe.iloc[i,:].sort_values(ascending=False)
        top_topic_name = ordered_row.index[0]
        count_pattern = re.compile("\d+")
        top_topic_number = count_pattern.search(top_topic_name).group()
        top_topic_list.append(int(top_topic_number))
    return top_topic_list


df_x["top_topic_number_lda8"] = top_topic_number_extractor(transformed_data_8)


topics_list = []
for i in range(0,8):
    if df_x["top_topic_number_lda8"][0] == i:
        topics_list.append(1)
    else:
        topics_list.append(0)


#turn topics dict into df for joining
top_topics_df = pd.DataFrame(topics_list).transpose().copy()


#tfidf vectorize the words in the training set:
tfidf = TfidfVectorizer(min_df=5, max_df=.95, lowercase=False, stop_words="english", ngram_range=(1,3))
tfidf.fit_transform(data_1500_kf["proper_title"])

#transform the input into tfidf
tfidf_title = tfidf.transform(df_x["proper_title"])


#make it into a df to join onto beginning data:
tfidf_title_df = pd.DataFrame(tfidf_title.todense(), 
                  columns=tfidf.get_feature_names())


#make X to match model X (using X2 here to match the final model)
X2 = pd.concat([df_x[["age_in_2017", "sentiment_polarity", "sentiment_subjectivity", "number_of_words", "title_length"]], top_topics_df, tfidf_title_df, reading_level_dummies], axis=1)

#this makes a one-line csv that can then be put through the model
X2.to_csv("test_data/test titles/X2_Marking_Identityc_Maori_Tattoos_and_Cultural_History.csv", index=False, encoding="utf-8")


