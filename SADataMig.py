#Python program to preprocess and sort tweets before performing Sentiment Analysis using Text Blob
import csv
import pandas as pd
import numpy as np
from textblob import TextBlob

def writeToFile(file, outputArray):
    with open(file, mode='a') as tweets:
        tweetWriter = csv.writer(tweets, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        tweetWriter.writerow(outputArray)                               
        tweets.close()

def getHeader(file):
    header = ["Keyword", "User", "Screen_Name", "Created_At", "Tweet_Text", "Sentiment", "Objectivity"]
    writeToFile(file, header)

def getOutput(keyword, user, screen_name, created_at, tweet_text, sentiment, objectivity):
    output = [keyword, user, screen_name, created_at, tweet_text, sentiment, objectivity]
    return output

preprocessFile = "preprocessing_list.csv"
dataFile = "Twitter Data Master.csv"
outputFile = "Twitter Data Master - Complete.csv"
researchData = "Twitter Data Master - Research.csv"

df = pd.read_csv(dataFile)# Read .csv file into dataFrame
dg = pd.read_csv(preprocessFile)#read file to preprocess tweets

# print(dg)

# print(df.Keyword)# print all data in df.Keyword

#-------------------------------------------------------------------------------------
#df Columns:    Keyword, Tweet, Senitment, Objectivity
#dg Columns:    Keyword, User, Screen_Name, Created_At, Tweet_Type, Tweet_Text, category, Note, Tweet Text, Tweet ID, Retweet Count, Favorite Count, Language, Geo Location, Profile Location
# print(dg.columns.values) # Get each column title
#----------------------------------------------------------
# add Main file columns to parallel array and rerun the SA
# find values of geo location at the same time

# print(df)
getHeader(outputFile)

rowCount = 0
elementCount = 0
replacementString = ""

for row in df.Tweet_Text:
    tweetString = row
    
    elementCount = 0
    for element in dg.Original_Term:
        # print(element)
        if element in row:

            print("\n" + element + " || " + row + " || " + dg.New_Term[elementCount])
            print(tweetString.replace(element, dg.New_Term[elementCount]))
            print("Term replaced")
            #replace original term with new term
            break
            # print("Term match found: " + str(element) + row)
        elementCount += 1

    wiki = TextBlob(tweetString) # run Sentiment Analysis on each row of the CSV

    # print(df.Keyword[rowCount] + " || " + df.Screen_Name[rowCount] + " || " + df.Created_At[rowCount])
    # print(wiki.sentiment.polarity) # Get tweet sentiment analysis
    try:
        writeToFile(outputFile, getOutput(df.Keyword[rowCount], df.User[rowCount], df.Screen_Name[rowCount], df.Created_At[rowCount], tweetString, wiki.sentiment.polarity, wiki.sentiment.subjectivity))
    except KeyError:
        print("Error Occurred")    
    rowCount += 1

print("Finished...")
