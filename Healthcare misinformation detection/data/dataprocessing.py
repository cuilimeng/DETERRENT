import pandas as pd
import json
import csv

df = pd.read_csv('Diabetes_cleaned.csv')
entitydf = pd.read_csv("entity2id.csv")
print(df.head())

entitydf["entity_name"] = entitydf["entity_name"].str.lower()
dictentity = pd.Series(entitydf.id.values,index=entitydf.entity_name).to_dict()
print(dictentity)

"""
dictrelation = []

with open("output.csv", encoding='utf-8-sig') as f:
    outputs = f.read().splitlines()

for output in outputs:

    relations = output.lower().split(',')
    l = dictentity[relations[0]]
    r = dictentity[relations[2]]
    dictrelation.append([l, int(relations[1]), r])

file = open('output.json','w',encoding='utf-8')
json.dump(dictrelation,file,ensure_ascii=False)

"""



file = open('train.json','w',encoding='utf-8')

dicttoken = {}

i = 24805

data = []
for index, row in df.iterrows():
   news = {"text": row['content'],
           "title": row['title'],
           "url": row['url'],
          "text_mentioned_tokens": [],
          "text_mentioned_entities": [],
          "label": None}
   for key in dictentity.keys():
       if key.lower() in row['content']:
           news["text"].replace(key,key.replace(' ', '_'))
           news['text_mentioned_entities'].append(dictentity[key])
   news['label'] = row['label']
   data.append(news)
json.dump(data,file,ensure_ascii=False)

file = open('train.json','r',encoding='utf-8')
data = json.load(file)

file = open('test.json','w',encoding='utf-8')
for news in data:
    words = news["text"].split(' ')
    for word in words:
        if word not in dictentity.keys():
            if word not in dicttoken.keys():
                i+=1
                dicttoken[word] = i
                news["text_mentioned_tokens"].append(i)
            else:
                news["text_mentioned_tokens"].append(dicttoken[word])
        else:
            news["text_mentioned_tokens"].append(dictentity[word.replace('_', ' ')])

json.dump(data,file,ensure_ascii=False)

w = csv.writer(open("token2id.csv", "w"))
for key, val in dicttoken.items():
    w.writerow([val, key])
