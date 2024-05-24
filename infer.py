
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split


from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import random_split
import datasets

import string
from collections import deque
from random import sample

def preprocess_text(text):
    # to lower case
    text = text.lower()
    # remove links
    text = re.sub('https:\/\/\S+', '', text) 
    # # remove punctuation
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    # remove next line     
    text = re.sub(r'[^ \w\.]', ' ', text) 
    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    return text

def count_token(tokenizer, text: str):
    return len(tokenizer.encode(text))

def split_text(tokenizer, text, MAX_LEN = 300):
    text_list = text.split('.')
    cnt_token = []
    queue = deque()
    res = []
    index = 0
    
    while index < len(text_list):
        text = text_list[index]
        n = count_token(tokenizer, text)
        
        # if len queue = 0 -> append
        if len(queue) == 0:
            queue.append(text)
            index += 1
            cnt_token.append(n)
        
        # if current string's tokens > MAX_LEN -> pop
        while sum(cnt_token) + n > MAX_LEN:
            if len(queue) == 0:
                queue.append(text)
                cnt_token.append(n)
                temp += text
                break   
            queue.popleft()
            cnt_token = cnt_token[1:]

        
        # if current string's tokens < MAX_LEN -> append
        if index == len(text_list): break
        text = text_list[index]
        n = count_token(tokenizer, text)
        while sum(cnt_token) + n < MAX_LEN:
            if index >= len(text_list):
                break
            
            queue.append(text)
            cnt_token.append(n)
            
            index += 1
            if index == len(text_list): break
            text = text_list[index]
            n = count_token(tokenizer, text)
            
            
        res.append(" ".join(queue))
    res = list(map(preprocess_text, res))
    return sample(res, int(len(res) * 0.9))

class CustomDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = split_text(tokenizer, text)
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding= "max_length",
            # pad_to_max_length=True,
            truncation = True,
            return_token_type_ids=True,
            return_tensors = 'pt',
        )
        
        # inputs = tokenizer.encode_plus(text, 
        #                 add_special_tokens=True, 
        #                 return_token_type_ids= True,
        #                 max_length=MAX_LEN, 
        #                 padding="max_length", 
        #                 return_tensors='pt') 
        ids = inputs['input_ids'].squeeze()
        mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()

        
        return {
            'ids': ids.clone().detach().long(),
            'mask': mask.clone().detach().long(),
            'token_type_ids': token_type_ids.clone().detach().long(),
        }

class CustomPandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = df
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding= "max_length",
            # pad_to_max_length=True,
            truncation = True,
            return_token_type_ids=True,
            return_tensors = 'pt',
        )
        
        # inputs = tokenizer.encode_plus(text, 
        #                 add_special_tokens=True, 
        #                 return_token_type_ids= True,
        #                 max_length=MAX_LEN, 
        #                 padding="max_length", 
        #                 return_tensors='pt') 
        ids = inputs['input_ids'].squeeze()
        mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()

        
        return {
            'ids': ids.clone().detach().long(),
            'mask': mask.clone().detach().long(),
            'token_type_ids': token_type_ids.clone().detach().long(),
        }
class Toxic_RoBERTa(torch.nn.Module):
    def __init__(self, mode= None):
        super(Toxic_RoBERTa, self).__init__()
        self.back_bone = AutoModelForSequenceClassification.from_pretrained("bstrai/multilingual-toxic-xlm-roberta")
        if mode == 'transfer':
            for param in self.back_bone.parameters():
                param.requires_grad = False
        else:
            for param in self.back_bone.parameters():
                param.requires_grad = True
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.back_bone(ids, attention_mask = mask, token_type_ids= token_type_ids, return_dict=False)[0]
        output = self.fc(output_1)
        return output
    
class Model:
    def __init__(self, weight= 'toxic_detection/ckpt/model-fine-tune-done2.pth', MAX_LEN = 300, device= 'cpu'):
        self.model = Toxic_RoBERTa()
        self.tokenizer = AutoTokenizer.from_pretrained("bstrai/multilingual-toxic-xlm-roberta")
        self.model.load_state_dict(torch.load(weight))
        self.model.eval()
        self.device= device
        self.max_len = MAX_LEN
        self.model.to(self.device)
    def predict_single_text(self, text):
        # self.model.to(self.device)
        inputs = self.tokenizer.encode_plus(text, 
                        add_special_tokens=True, 
                        return_token_type_ids= True,
                        max_length= 300, 
                        padding="max_length", 
                        return_tensors='pt') 
    
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        
        with torch.no_grad():      
            outputs = self.model(ids, mask, token_type_ids)
        return torch.sigmoid(outputs).clone().cpu().detach().numpy().tolist()
    
    def predict(self, text):
        dataset = CustomDataset(text= text, tokenizer= self.tokenizer, max_len= self.max_len)
        data_loader = DataLoader(dataset, batch_size= 16, shuffle= False)
        
        # self.model.to(self.device)
        self.model.eval()
        
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(data_loader, 0):
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                
                
                outputs = self.model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(outputs).clone().cpu().detach().numpy().tolist())   
                # print('a')  
                       
        return np.array(fin_outputs).squeeze().max()
    
    def predict_dataframe(self, df):
        dataset = CustomPandasDataset(df, tokenizer= self.tokenizer, max_len= self.max_len)
        data_loader = DataLoader(dataset, batch_size= 16, shuffle= False)
        
        # self.model.to(self.device)
        self.model.eval()
        
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(data_loader, 0):
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                
                
                outputs = self.model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(outputs).clone().cpu().detach().numpy().tolist())   
                # print('a')  
                       
        return np.array(fin_outputs).squeeze()
            
def main():
    # # Test 1: Load model
    # model = Toxic_RoBERTa(mode='transfer')
    # print(model)
    
    # #Test 2: predict single text
    # device = 'cuda:2'
    # model = Model(device=device)
    # text = 'that stupid guy told me that I am a bad person. I am so sad'
    # print(model.predict_single_text(text))
    
#     # Test 3: count tokens
#     tokenizer = AutoTokenizer.from_pretrained("bstrai/multilingual-toxic-xlm-roberta")
    text = '''Preface
Making Object,Oriented Design Accessible
This book is an introduction to object-oriented design and design patterns at an
elementary level. It is intended for students with at least one semester ofprogramming in an object-oriented language such as Java or C++.
I wrote this book to solve a common problem. When students first learn an
object-oriented programming language, they cannot be expected to instantly master object-oriented design. Yet, students should learn the principles of object-oriented design early enough to put them to work throughout the computer science
curriculum.
This book is suitable for a second or third course in computer science-no background in data structures is required, and students are not assumed to have experience with developing large software systems. Alternatively, the book can be used
as a companion text in a course in software engineering. (If you need a custom
version of this book for integration into another course, please contact your Wiley
sales representative.)
This second edition is fully updated for Java 5.0, including
• the use ofgeneric collections and the "for each" loop
• a detailed discussion of parameterized type constraints
• auto-boxing and varargs methods, particularly in the reflection API
• multithreading with the j ava . uti 1 . concu r rent package
Integration of Design Patterns
The most notable aspect of this book is the manner in which the coverage of
design patterns is interwoven with the remainder of the material. For example,
• Swing containers and components motivate the COMPOSITE pattern.
• Swing scroll bars motivate the DECORATOR pattern, and Swing borders are
examined as a missed opportunity for that pattern.
• Java streams give a second example ofthe DECORATOR pattern. Seeing the
pattern used in two superficially different ways greatly clarifies the pattern
concept.'''
#     print(count_token(tokenizer, text))

    # # Test 4: test preprocess text
    # print(preprocess_text(text))

    # # Test 5: split text
    # text_split = split_text(tokenizer, text)
    # for i in text_split:
    #     print(f'text: {i}')
    #     print(f'Nums of tokens: {count_token(tokenizer, i)}')
    #     print('--------------------------------------------------------------------------')
    
    ## Test 6: Test dataset
    # dataset = CustomDataset(text, tokenizer, 300)
    # print(len(dataset))
    # for item in dataset:
    #     print(item['ids'])
    #     break
    
    # Test 7: Test model with long text
    device = 'cuda:2'
    model = Model(device=device)
    print(model.predict(text))
if __name__ == '__main__':
    main()
    
    

