#!/usr/bin/env python
# coding: utf-8

# In[19]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from nltk import word_tokenize
import numpy as np
import random


# In[5]:


questions = ["hi", "where am i", "who are you", "help", "i need tips", "what is your age", "can you give me some tips", 
            "are you 25?"]
responses = {
    "greet":["Hi, there", "Hello!", "Welcome back!"], 
    "place":["Hey you are in Kavai!!", "You are in Rinaldo's Mansion"],
    "name":["I am Rinaldo Nikilson", "Rinaldo Nikilson"],
    "help":["Please tell me how can i help you!!", "I am here to help you sir!"],
    "tips":["Please don't look at the keyboard while typing!!", "Please plug your power while heavy task in laptop"],
    "age":["I am 22", "My age is 22!!"]
    
}
key_responses = ["greet", "place", "name", "help", "tips", "age", "tips", "age"]


# In[6]:


bag_of_words = []
for sentence in questions:
    for word in word_tokenize(sentence):
        if not (word in bag_of_words):
            bag_of_words.append(word)
bag_of_words = sorted(bag_of_words)


# In[7]:


print(bag_of_words)


# In[8]:


questions_numbers = []
for sentence in questions:
    numbered_words = [0 for _ in bag_of_words]
    for word in word_tokenize(sentence):
        loc = bag_of_words.index(word)
        numbered_words[loc] += 1
    questions_numbers.append(numbered_words)


# In[9]:


print(len(questions_numbers[0]))


# In[10]:


bag_of_keys = list(responses)
print(bag_of_keys)
response_numbers = []


# In[11]:


for key in key_responses:
    num_list = [0 for _ in bag_of_keys]
    for n, cmp in enumerate(bag_of_keys):
        if cmp == key:
            num_list[n] = 1
    response_numbers.append(num_list)
print(response_numbers)


# In[12]:


x_train = np.array(questions_numbers)
y_train = np.array(response_numbers)
print(x_train.shape, y_train.shape)
print((questions_numbers[0]), response_numbers[0])


# In[13]:


model = Sequential()
model.add(Dense(len(questions_numbers[0]), input_dim=len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(questions_numbers[0]), activation='relu'))
model.add(Dense(len(response_numbers[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[14]:


history = model.fit(x_train, y_train, epochs=200)


# In[15]:


def get_bag_of_words(text, bag_of_words):
    numbered_words = [0 for _ in bag_of_words]
    for word in word_tokenize(text):
        if word in bag_of_words:
            loc = bag_of_words.index(word)
            numbered_words[loc] += 1
    return numbered_words


# In[22]:


while True:
    text = input("You : ")
    if text == "quit":
        break
    x_test = get_bag_of_words(text, bag_of_words)
    x_test = np.array(x_test)
    x_test = x_test.reshape(1, len(x_test))
    # print(x_test)
    y_pred = model.predict(x_test)
    y_pred = (y_pred.argmax())
    pred_resp = key_responses[y_pred]
    output = responses[pred_resp]
    output = random.choice(output)
    print(f"Rinaldo : {output}")


# In[ ]:





# In[ ]:




