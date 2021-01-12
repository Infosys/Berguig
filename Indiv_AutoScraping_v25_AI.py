

# -*- coding: utf-8 -*-
"""
Copyright 2018 Infosys Ltd.
Use of this source code is governed by MIT license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

@author: zineb.mezzour , mohan.ponduri
# -*- coding: utf-8 -*-
"""

#%%


import xml.dom.minidom
import pandas as pd
import requests
import datetime as DT
from dateutil.parser import parse
import win32com.client as win32
import newspaper
from newspaper import Article
import nltk
nltk.download('all')
from TwitterSearch import TwitterSearchOrder
from TwitterSearch import TwitterUserOrder
from TwitterSearch import TwitterSearchException
from TwitterSearch import TwitterSearch
from bs4 import BeautifulSoup as bs
import urllib3
import xmltodict
import traceback2 as traceback
import re

import warnings
import contextlib
from urllib3.exceptions import InsecureRequestWarning

import Algo_4



#%%

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

keywords=[]
companies_names=[]
days_count=[]
emails=[]
persons_names=[]
connections=[]
companies_ids=[]
relevantsubject=[]
twitter_list=[]
main_list=[]
log_date=[]

def main():


#COMPANIES.XML
    
    doc_comp = xml.dom.minidom.parse("companies.xml");
    
#    print(doc.nodeName)
#    print(doc.firstChild.tagName)

    companies=doc_comp.getElementsByTagName("company")
    
   

    for company in companies:  
        #print(company.getElementsByTagName("name"))
        company_name=company.getElementsByTagName("c_name")[0].childNodes[0]
        companies_names.append(company_name.nodeValue)
        
        keyword=company.getElementsByTagName("keyword")
        x=[]
        for word in keyword:
            x.append(word.childNodes[0].nodeValue)
        keywords.append(x)
    
        company_id=company.getElementsByTagName("c_id")[0].childNodes[0]
        companies_ids.append(company_id.nodeValue)
        
        twitter=company.getElementsByTagName("twitter_name")[0].childNodes[0].nodeValue
        
        youtube=company.getElementsByTagName("youtube")[0].childNodes[0].nodeValue
        
        hashtag=company.getElementsByTagName("hashtag")
        
        z=[]
        for word in hashtag:
            z.append(word.childNodes[0].nodeValue)
        
        twitter_list.append([twitter,z])
        
        main_list.append([company_name.nodeValue,x,twitter,z,youtube])
        

    
#NEW DATE
    doc_log = xml.dom.minidom.parse("log.xml");
    
    log_date.append(doc_log.getElementsByTagName('day')[0].childNodes[0].nodeValue)
    
       

#PEOPLE.XML
    
    doc = xml.dom.minidom.parse("people_v2.xml");
    
    #print(doc.nodeName)
    #print(doc.firstChild.tagName)

    person=doc.getElementsByTagName("person")
    
    
    for info in person:  
#        print(company.getElementsByTagName("name"))
        person_name=info.getElementsByTagName("p_name")[0].childNodes[0]
        #print(person_name)
        persons_names.append(person_name.nodeValue)
        
        email=info.getElementsByTagName("email")[0].childNodes[0]
        emails.append(email.nodeValue)
        
        
        grouped_company=info.getElementsByTagName("group")
        
        group=[]
        for g in grouped_company:
            group_name=g.getElementsByTagName("g_name")[0].childNodes[0]
            #group.append(group_name.nodeValue)
            
            comp_name=g.getElementsByTagName("comp_id")
            comp=[]
            for c in range(len(comp_name)):
                comp.append(comp_name[c].childNodes[0].nodeValue)
            group.append([group_name.nodeValue,comp])
        
        #connections.append(group)
        
        
        single_companies=info.getElementsByTagName("single")[0]
        cs_name=single_companies.getElementsByTagName("comp_id")
        single_comp=[]
        for s in range(len(cs_name)):
            single_name=cs_name[s].childNodes[0].nodeValue
            single_comp.append(single_name)
        
        group.append(single_comp)
        
        connections.append(group)
        

#Keywords.XML        
        
    doc_words = xml.dom.minidom.parse("keywords_list.xml");
    #print(doc_date.nodeName)
    
    for i in range(len(doc_words.getElementsByTagName('word'))):
        word=doc_words.getElementsByTagName('word')[i]
   
        l=word.childNodes[0].nodeValue
        relevantsubject.append(l)    
    
    
        

if __name__ == "__main__":
    main();




#%%
urls=[]
current_companies=[]
datasets={}
API_KEY = ''

def content():            
            
 
    today = DT.date.today()
#    days_ago = today - DT.timedelta(days=int(days_count[0]))
    
    todayf = today.strftime("%Y-%m-%d")
#    days_agof = days_ago.strftime("%Y-%m-%d")
    
    
    #URLS
    
    url = 'https://newsapi.org/v2/everything?q='
    url_p2='&from='+log_date[0]+'&to='+todayf+'+&sortBy=publishedAt&language=en&apiKey='+ API_KEY
    
    for company in range(len(keywords)):
    #    print(company)
    #    print(len(company))
              
        if len(keywords[company]) == 0 :
            print('no keywords given')
            
        if len(keywords[company]) > 1 :
            new_url = url + keywords[company][0]
            for i in range(1,len(keywords[company])):
                new_url = new_url + "%20AND%20"+ keywords[company][i]
            
            final_url = new_url + url_p2            
        
        else:
            final_url= url + keywords[company][0] + url_p2
            
    #    print(url)
        urls.append(final_url)
        
    
    # Build df with article info + create excel sheet
    count = 0  
#    current_companies=[]
#    datasets={}
    
    
    for url in urls:
        
            JSONContent = requests.get(url).json()
            #content = json.dumps(JSONContent, indent = 4, sort_keys=True)
            
    
            article_list = []
            
            for i in range(len(JSONContent['articles'])):
                article_list.append([JSONContent['articles'][i]['source']['name'],
                                     JSONContent['articles'][i]['title'],
                                     JSONContent['articles'][i]['publishedAt'],
                                     JSONContent['articles'][i]['url']
                                     ])
            
            #print(article_list)
            
            if article_list != []:
                datasets[companies_names[count]]= pd.DataFrame(article_list)
                datasets[companies_names[count]].columns = ['Source/User','Title/Tweet','Date','Link']
                                
                datasets[companies_names[count]]['Date']=datasets[companies_names[count]]['Date'].str.replace('T',' ')
                datasets[companies_names[count]]['Date']=datasets[companies_names[count]]['Date'].str.replace('Z','')
                datasets[companies_names[count]]['Date']=datasets[companies_names[count]]['Date'].str.split(expand=True)

                for i in range(len(datasets[companies_names[count]]['Date'])):
                    
                    datasets[companies_names[count]]['Date'][i]=parse(datasets[companies_names[count]]['Date'][i])
                    datasets[companies_names[count]]['Date'][i]=datasets[companies_names[count]]['Date'][i].date()
                    #datasets[companies_names[count]]['Date'][i]=datasets[companies_names[count]]['Date'][i].str.split(expand=True)
                #ds = '2012-03-01T10:00:00Z' # or any date sting of differing formats.
                #date = parser.parse(ds)

                #datasets[companies_names[count]]['Date']=pd.to_datetime(datasets[companies_names[count]]['Date'])

                #print(datasets[companies_names[count]])
                
                current_companies.append(companies_names[count])
                            
                count=count+1 
             
            
            else:
                None
                count=count+1 
                


content()





duplicate_df=[]


def duplicate_check():
    for article in datasets:
        d=datasets[article][datasets[article].duplicated(['Title/Tweet'],keep='first')==True]
        print(d)
        
        if d.empty == False:
            duplicate_df.append(d)
        else:
            None
        
        
        #duplicate_article.append(d)
     
        #duplicate_article = duplicate_article.concat([duplicate_article,d], axis=0)
        #print(d)    
                           
duplicate_check()



def duplicate_drop():
    for article in datasets:
        datasets[article]=datasets[article].drop_duplicates(['Title/Tweet'],keep='first')
        datasets[article]=datasets[article].reset_index()
        datasets[article]=datasets[article].drop(['index'], axis=1)
        
                           
duplicate_drop()



#%%




def Scoring():
    

    for a in datasets:
        
        try:
            
            datasets[a].insert(0,'Category','Article')
            datasets[a].insert(1,'Company',str(a))
            
            datasets[a].insert(3,'Keywords','none')
            
            datasets[a].insert(4,'Subjects/Views','none')
            for i in range(len(datasets[a]['Link'])):
                r=[]
                article = Article(datasets[a]['Link'][i])
                article.download()
               
                
                article.html
                article.parse()
                txt=article.text.encode('ascii','ignore').decode('ascii')
                #f=requests.get(datasets[article]['Link'][i])
                #txt=f.text.encode('ascii','ignore').decode('ascii')
                txt=txt.lower()
                #total_word= wordcounter(txt).get_word_count()
                
                for word in relevantsubject:
                    result=txt.count(word)            
                    if result != 0:
                        r.append(word +'('+ str(txt.count(word)) +')')
                    else:
                        None
                        
               # relevanceLink.append(r) 
                r=', '.join(word for word in r)
                
                if r != []:
                    datasets[a]['Subjects/Views'][i]=str(r + ' (totalWords:'+ str(len(txt.split()))+')')
                
                else:
                    datasets[a]['Subjects/Views'][i]=str('None')
                
                
                article.nlp()
                k=', '.join(keyword for keyword in article.keywords)
                datasets[a]['Keywords'][i]=str(k)
                
        except newspaper.article.ArticleException:
            None
            
            
            #k= []
            #for keyword in article.keywords:
            #    k.append[keyword]
            #    k=', '.join(keyword for keyword in k)
            #    datasets[a]['Keywords'][i]=str(k)
            
            
            
            
                

Scoring()       
 
 #   datasets[article]




#%% Formatting
 
 
companies_dic=dict(zip(companies_names, companies_ids))
people_comp_dic=dict(zip(persons_names, connections))
people_email_dic=dict(zip(persons_names, emails))


Subject = pd.DataFrame(relevantsubject)
Subject.columns=['Subject Interest']
Companies = pd.DataFrame(companies_names)
Companies.columns=['Companies Interest']
CS = pd.concat([Subject, Companies], axis=1)
CS.fillna('',inplace=True)



MainDF=pd.DataFrame(main_list)
MainDF.columns=['company','keywords','twitter','hashtag','youtube']




#import re 

def Find(string):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F])|(?:%[0-9a-fA-F]|[$-_@.&+]|[!*\(\), ]|[0-9a-fA-F]))+', string) 
    return url 

     


#%%

tweets_datasets={}
tw_current_companies=[]
today = DT.date.today()
#days_ago = today - DT.timedelta(days=int(days_count[0]))
new_date = parse(log_date[0]).date()

def Tweets():
    

    try:
        

        max_feeds=10
        tso = TwitterSearchOrder() # create a TwitterSearchOrder object
        tso.set_language('en') 
        tso.set_include_entities(False) # and don't give us all those entity information
        tso.set_until(new_date)
        tso.arguments.update({'tweet_mode':'extended'})
        tso.arguments.update({'truncated': 'False' })
    
    
        ts = TwitterSearch(
            consumer_key = '',
            consumer_secret = '',
            access_token = '',
            access_token_secret = '',
            proxy='http://proxy_address'
            )
        
        for c in range(len(MainDF)):
            count=0
            
            #kw=[MainDF['twitter'][c]]
            #for h in MainDF['hashtag'][c]:
            #    kw.append(h)
            
            tso.set_keywords(MainDF['hashtag'][c])
            tweets_list=[]
            
            
            tuo = TwitterUserOrder(MainDF['twitter'][c])
#            tuo.set_language('en') 
            tuo.set_include_entities(False) # and don't give us all those entity information
#            tuo.set_until(days_ago)
#            tuo.set_count(15)
            tuo.arguments.update({'tweet_mode':'extended'})
            tuo.arguments.update({'truncated': 'False' })
            
    
    
            #for tweet in ts.search_tweets_iterable(tso):
            #    print(tweet)
            #    tweets_list.append([tweet['user']['screen_name'],tweet['full_text']])
            
    
            
            for tweet in ts.search_tweets_iterable(tso):
                if 'retweeted_status' in tweet:
                    None
                    #tweets_list.append([tweet['user']['screen_name'],tweet['retweeted_status']['full_text'],'Retweet of ' + tweet['retweeted_status']['user']['screen_name']])
                else:
                    links=Find(tweet['full_text'])
                    links=', '.join(link for link in links)
                    #print(tweet)
                    tweets_list.append([MainDF['company'][c],tweet['user']['screen_name'],tweet['full_text'],tweet['created_at'],links])
            
                    
            
            for tweet in ts.search_tweets_iterable(tuo):
                if tweet['lang'] != 'en':
                    #print(tweet)
                   None
                else:
                    
                   # print(tweet)
                    links=Find(tweet['full_text'])
                    links=', '.join(link for link in links)
                            
                    tweets_list.append([MainDF['company'][c],tweet['user']['screen_name'],tweet['full_text'],tweet['created_at'],links])
                    count=count+1
                            
                    if count == max_feeds:
                        break
                    
            
            if tweets_list != []:
                tweets_datasets[MainDF['company'][c]]= pd.DataFrame(tweets_list)
                tweets_datasets[MainDF['company'][c]].columns = ['Company','Source/User','Title/Tweet','Date','Link']
                tweets_datasets[MainDF['company'][c]].insert(0,'Category','Twitter')
                
                for i in range(len(tweets_datasets[MainDF['company'][c]]['Date'])):
                    
                    tweets_datasets[MainDF['company'][c]]['Date'][i]=parse(tweets_datasets[MainDF['company'][c]]['Date'][i])
                    tweets_datasets[MainDF['company'][c]]['Date'][i]=tweets_datasets[MainDF['company'][c]]['Date'][i].date()
                
                
                    #print(datasets[companies_names[count]])
                    
                tw_current_companies.append(MainDF['company'][c])
                                
                 
            
            else:
                None
                
              
            #tweets_list.append()
            #print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
    
    except TwitterSearchException as e: # take care of all those ugly errors if there are some
        print(e)

with no_ssl_verification():
    Tweets()


#%% Filters only for todays 
for comp in tweets_datasets:
    tweets_datasets[comp]=tweets_datasets[comp].loc[tweets_datasets[comp]['Date'] >= new_date]    

for comp in list(tweets_datasets.keys()):    
    if tweets_datasets[comp].empty == True:
        del tweets_datasets[comp]

#re-indexing
for comp in tweets_datasets:
   tweets_datasets[comp]=tweets_datasets[comp].reset_index()    
   tweets_datasets[comp]=tweets_datasets[comp].drop(['index'], axis=1)
 #tweets_datasets = tweets_datasets.loc[tweets_datasets[comp].empty == False]


#%%


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

Double_df=[]


for comp in tweets_datasets:
    
    for i in range(len(tweets_datasets[comp])):
        
        doubles=[]
        #doubles.append(comp)
        X =tweets_datasets[comp]['Title/Tweet'][i]
        X_list = word_tokenize(X) 
        
        sw = stopwords.words('english')  
        
        X_set = {w for w in X_list if not w in sw}  
        
        for n in range(len(tweets_datasets[comp])):
            Y =tweets_datasets[comp]['Title/Tweet'][n]
              
            # tokenization 
            Y_list = word_tokenize(Y) 
              
            # sw contains the list of stopwords 
           # sw = stopwords.words('english')  
            l1 =[];l2 =[] 
              
            # remove stop words from string 
            #X_set = {w for w in X_list if not w in sw}  
            Y_set = {w for w in Y_list if not w in sw} 
              
            # form a set containing keywords of both strings  
            rvector = X_set.union(Y_set)  
            for w in rvector: 
                if w in X_set: l1.append(1) # create a vector 
                else: l1.append(0) 
                if w in Y_set: l2.append(1) 
                else: l2.append(0) 
            c = 0
              
            # cosine formula  
            for i in range(len(rvector)): 
                    c+= l1[i]*l2[i] 
                    
            cosine = c / float((sum(l1)*sum(l2))**0.5) 
            print(tweets_datasets[comp]['Title/Tweet'][n])
            print("similarity: ", cosine) 
            
            if (Y == X)== True:
                
                #None
                print('Same')
            else:
                
                if 0.80 <= cosine <= 0.99 :
                    print('Yes!')
                    doubles.append(tweets_datasets[comp].iloc[[n]])
                    #d=tweets_datasets[comp][tweets_datasets[comp]['Title/Tweet'][n]]
                    #doubles.append(d)
                else:
                    None
    
    if doubles != []:
        d=pd.concat(doubles)
        d=d.reset_index()
        d=d.drop(['index'],axis=1)
        Double_df.append(d)
    else:
        None

        
def drop_similar():
    for comp in tweets_datasets:
        for i in range(len(Double_df)):
            for n in range(len(Double_df[i])):
                for r in range(len(tweets_datasets[comp].copy())):
                        
                    if Double_df[i]['Title/Tweet'][n] != tweets_datasets[comp]['Title/Tweet'][r]:
                        None
                    else:
                        tweets_datasets[comp]=tweets_datasets[comp].drop(r)
        tweets_datasets[comp]=tweets_datasets[comp].reset_index()
        tweets_datasets[comp]=tweets_datasets[comp].drop(['index'], axis=1)
            
drop_similar()
#%%

tw_duplicate_df=[]


def tw_duplicate_check():
    try:
        for article in tweets_datasets:
            d=tweets_datasets[article][tweets_datasets[article].duplicated(subset=['Title/Tweet'],keep='first')==True]
            print(d)
            
            if d.empty == False:
                tw_duplicate_df.append(d)
            else:
                None
    except:
        None
             
                           
tw_duplicate_check()



def tw_duplicate_drop():
    if tw_duplicate_df != []:
        
        for article in tweets_datasets:
            tweets_datasets[article]=tweets_datasets[article].drop_duplicates(subset=['Title/Tweet'],keep='first', inplace=True)
            tweets_datasets[article]=tweets_datasets[article].reset_index()
            tweets_datasets[article]=tweets_datasets[article].drop(['index'], axis=1)
     
    else:
         None
                           
tw_duplicate_drop()





#%%

def Scoring_Tweet():
     
    
    for a in tweets_datasets:
        
        #datasets[a].insert(0,'Company',str(a))
        
        
        tweets_datasets[a].insert(3,'Subjects/Views','none')
        for i in range(len(tweets_datasets[a]['Title/Tweet'])):
            r=[]

            txt=tweets_datasets[a]['Title/Tweet'][i].encode('ascii','ignore').decode('ascii')
            #f=requests.get(datasets[article]['Link'][i])
            #txt=f.text.encode('ascii','ignore').decode('ascii')
            txt=txt.lower()
            #total_word= wordcounter(txt).get_word_count()
            
            for word in relevantsubject:
                result=txt.count(word)            
                if result != 0:
                    r.append(word +'('+ str(txt.count(word)) +')')
                else:
                    None
                    
           # relevanceLink.append(r) 
            r=', '.join(word for word in r)
            
            if r != []:
                tweets_datasets[a]['Subjects/Views'][i]=str(r + ' (totalWords:'+ str(len(txt.split()))+')')
            
            else:
                tweets_datasets[a]['Subjects/Views'][i]=str('None')
    
            
Scoring_Tweet()





#%%
general_df = {}

general_df = tweets_datasets.copy()


for n in datasets:
    if n in general_df:
        general_df[n]=pd.concat([datasets[n],general_df[n]], axis=0, sort=False)
    else:
        general_df.update({str(n):datasets[n]})
        


for comp in general_df:
   general_df[comp]=general_df[comp].reset_index()    
   general_df[comp]=general_df[comp].drop(['index'], axis=1)            
            
   
#%%
Youtube_dataset ={}
base = "https://www.youtube.com/user/{}/videos"
from textblob import TextBlob

#qstring = "snowflakecomputing"
    
for i in range(len(MainDF)):
    
    qstring= MainDF['youtube'][i]
    with no_ssl_verification():
        r = requests.get(base.format(qstring) )   
    
    page = r.text
    soup=bs(page,'html.parser')
    
    vids= soup.findAll('a',attrs={'class':'yt-uix-tile-link'})
    duration=soup.findAll('span',attrs={'class':'accessible description'})
    date=soup.findAll('ul',attrs={'class':'yt-lockup-meta-info'})
    
    
    videolist=[]
    for v in vids:
        tmp = 'https://www.youtube.com' + v['href']
        videolist.append([v['title'],tmp])
    
    infos=[]  
    for d in date:
        x=d.findAll('li')
        infos.append([x[0].text,x[1].text])
    
    
    youtubeDF=pd.DataFrame(videolist)
    infosDF=pd.DataFrame(infos)

    youtubeDF=pd.concat([youtubeDF,infosDF],axis=1)
    #print(youtubeDF)
    
    if youtubeDF.empty == False :
        

        youtubeDF.columns=['Title/Tweet','Link','Subjects/Views','Date']
        youtubeDF.insert(0,'Company',str(MainDF['company'][i]))
        youtubeDF.insert(0,'Category','youtube')
        youtubeDF.insert(2,'Source/User',base.format(qstring))
        last = youtubeDF.loc[youtubeDF['Date']=='1 day ago']
        last['Date']=last['Date'].replace('1 day ago',log_date[0], regex=True)
        
        
        if last.empty == False:
                
            Youtube_dataset[MainDF['company'][i]]=pd.DataFrame(last)    
                
        else:
            None
    else:
        None   
#%% 
        
#re-indexing
for comp in Youtube_dataset:
   Youtube_dataset[comp]=Youtube_dataset[comp].reset_index()    
   Youtube_dataset[comp]=Youtube_dataset[comp].drop(['index'], axis=1)
                       
#%%        
for i in list(Youtube_dataset.keys()):
   
    for n in range(len(Youtube_dataset[i])):
        
        if TextBlob(Youtube_dataset[i]['Title/Tweet'][n]).detect_language() != 'en':
            Youtube_dataset[i]=Youtube_dataset[i].drop(n)
            Youtube_dataset[i]=Youtube_dataset[i].reset_index()
            Youtube_dataset[i]=Youtube_dataset[i].drop(['index'], axis=1)
        else:
            None
                       
  #%%          
for comp in list(Youtube_dataset.keys()):    
    if Youtube_dataset[comp].empty == True:
        del Youtube_dataset[comp]     
#%%
#re-indexing
for comp in Youtube_dataset:
   Youtube_dataset[comp]=Youtube_dataset[comp].reset_index()    
   Youtube_dataset[comp]=Youtube_dataset[comp].drop(['index'], axis=1)

        
#%%

for n in Youtube_dataset:
    if n in general_df:
        general_df[n]=pd.concat([Youtube_dataset[n],general_df[n]], axis=0, sort=False)
    else:
        general_df.update({str(n):Youtube_dataset[n]})
        

for comp in general_df:
   general_df[comp]=general_df[comp].reset_index()    
   general_df[comp]=general_df[comp].drop(['index'], axis=1)          
        

   
#%% ITUNES
urls_itunes=[]
itunes_content={}

url_1='https://itunes.apple.com/search?lang=en_us&term='   

for i in range(len(companies_names)):
    url_final= url_1 + companies_names[i]
    #print(url_final)
    urls_itunes.append(url_final)
    

i = 0
    
for url in urls_itunes:
    
    response = requests.get(url)

    content = response.json()
    
    podcast_list=[]
    
    for n in range(len(content['results'])):
        c = content['results'][n]
        #print(content['results'][n].items())
        if content['results'][n]['wrapperType'] == 'audiobook':
            podcast_list.append(['audiobook',companies_names[i],c['artistName'],c['collectionName'],c['releaseDate'],c['collectionViewUrl']])
        else:
        
            if content['results'][n]['kind'] == 'podcast':        
                podcast_list.append(['podcast',companies_names[i],c['artistName'],c['collectionName'],c['releaseDate'],c['trackViewUrl']])
            else:
                None
                
        
    if podcast_list != []:
        itunes_content[companies_names[i]] = pd.DataFrame(podcast_list)     
        itunes_content[companies_names[i]].columns =['Category','Company','Source/User','Title/Tweet','Date','Link']
        itunes_content[companies_names[i]]['Date']=itunes_content[companies_names[i]]['Date'].str.replace('T',' ')
        itunes_content[companies_names[i]]['Date']=itunes_content[companies_names[i]]['Date'].str.replace('Z',' ')
        itunes_content[companies_names[i]]['Date']=itunes_content[companies_names[i]]['Date'].str.split(expand=True)
        
        for d in range(len(itunes_content[companies_names[i]]['Date'])):
                    
            itunes_content[companies_names[i]]['Date'][d]=parse(itunes_content[companies_names[i]]['Date'][d])
            itunes_content[companies_names[i]]['Date'][d]=itunes_content[companies_names[i]]['Date'][d].date()  
        
        i = i +1
    
    else:
        i = i +1


#Only for the good period of time 
        
for comp in itunes_content:
    itunes_content[comp]=itunes_content[comp].loc[itunes_content[comp]['Date'] >= new_date]
 
for comp in list(itunes_content.keys()):    
    if itunes_content[comp].empty == True:
        del itunes_content[comp]

#re-indexing
for comp in itunes_content:
   itunes_content[comp]=itunes_content[comp].reset_index()    
   itunes_content[comp]=itunes_content[comp].drop(['index'], axis=1)
       
     
#%%
for n in itunes_content:
    if n in general_df:
        general_df[n]=pd.concat([itunes_content[n],general_df[n]], axis=0, sort=False)
    else:
        general_df.update({str(n):itunes_content[n]})
        

for comp in general_df:
   general_df[comp]=general_df[comp].reset_index()    
   general_df[comp]=general_df[comp].drop(['index'], axis=1)          
           
   

   

#%% RSS FEED
 
 
    
#import urlopen
url='https://blogs.gartner.com/gbn-feed/'
proxy = urllib3.ProxyManager('http://proxy_address')
http = urllib3.PoolManager()
response = proxy.request('GET', url)
try:
    data = xmltodict.parse(response.data)
except:
    print("Failed to parse xml from response (%s)" % traceback.format_exc())



RSSfeed=pd.DataFrame(data['rss']['channel']['item'])




for r in range(len(RSSfeed)):
    RSSfeed['pubDate'][r]=parse(RSSfeed['pubDate'][r])
    RSSfeed['pubDate'][r]=RSSfeed['pubDate'][r].date()
    
    
RSSfeed=RSSfeed.loc[RSSfeed['pubDate'] >= new_date] 
#RSSfeed=RSSfeed.loc[RSSfeed['pubDate'] >= parse('2019-09-12').date()]

RSSfeed=RSSfeed.drop(['guid','author','headshot','category'], axis=1)
    
    
#%%

def RSS_scoring():
    try:
        
        if RSSfeed.empty == True:
            None
        else:
            
         
            RSSfeed.insert(0,'Category','Gartner')
            #RSSfeed.insert(1,'Company','/')
            RSSfeed.insert(3,'Keywords','none')
            RSSfeed.insert(4,'Subjects/Views','none')
            
            
            for i in range(len(RSSfeed['link'])):
                article = Article(RSSfeed['link'][i])
                article.download()
                           
                            
                article.html
                article.parse()
                txt=article.text.encode('ascii','ignore').decode('ascii')
                            #f=requests.get(datasets[article]['Link'][i])
                            #txt=f.text.encode('ascii','ignore').decode('ascii')
                txt=txt.lower()
                            #total_word= wordcounter(txt).get_word_count()
                r=[]            
                for word in relevantsubject:
                    result=txt.count(word)              
                    
                    if result != 0:
                        r.append(word +'('+ str(txt.count(word)) +')')
                    else:
                        None                        
                           # relevanceLink.append(r) 
                r=', '.join(word for word in r)
                
                if r != []:
                    RSSfeed['Subjects/Views'][i]=str(r + ' (totalWords:'+ str(len(txt.split()))+')')
                    
                else:
                    RSSfeed['Subjects/Views'][i]=str('None')
                
                
                article.nlp()
                k=', '.join(keyword for keyword in article.keywords)
                RSSfeed['Keywords'][i]=str(k)
        
    except:
            
        None
        
RSS_scoring()
#%%

if RSSfeed.empty != True:
    RSSfeed=RSSfeed.loc[RSSfeed['Subjects/Views'].str.contains('data') == True]

    if RSSfeed.empty == True:
        RSSfeed = RSSfeed.drop(['Category', 'title', 'link', 'Keywords', 'Subjects/Views', 'description','pubDate'], axis=1)
    else:
        None     
else:
    RSSfeed = RSSfeed.drop([ 'title', 'link', 'description','pubDate'], axis=1)
    



#%%


for i in RSSfeed:
    RSSfeed[i].fillna(" ", inplace=True)


NEWS = RSSfeed['title'] + RSSfeed['Keywords'] + RSSfeed['Subjects/Views']
NEWS = NEWS.apply(Algo_4.clean_text)

NEWS_cv = Algo_4.cv.transform(NEWS)
NEWS_tfdi = Algo_4.tfidf_transformer.transform(NEWS_cv)
    

predictions_news = Algo_4.naive_bayes.predict(NEWS_cv)
predictions_news=pd.Series(predictions_news)
RSSfeed=pd.concat([RSSfeed,predictions_news], axis=1)
RSSfeed.rename(columns={0:'Relevant(1)/Irrelevant(0)'}, inplace=True)


    
        

#%%
for i in general_df:
    for n in general_df[i]:
        general_df[i][n].fillna(" ", inplace=True)


for i in general_df:
    if 'Keywords' in general_df[i]:
        NEWS = general_df[i]['Title/Tweet'] + general_df[i]['Source/User']+ general_df[i]['Keywords'] + general_df[i]['Subjects/Views']
    else:
        NEWS = general_df[i]['Title/Tweet'] + general_df[i]['Source/User'] + general_df[i]['Subjects/Views']

    NEWS = NEWS.apply(Algo_4.clean_text)

    NEWS_cv = Algo_4.cv.transform(NEWS)
    NEWS_tfdi = Algo_4.tfidf_transformer.transform(NEWS_cv)
    

    predictions_news = Algo_4.naive_bayes.predict(NEWS_cv)
    predictions_news=pd.Series(predictions_news)
    general_df[i]=pd.concat([general_df[i],predictions_news], axis=1)
    general_df[i].rename(columns={0:'Relevant(1)/Irrelevant(0)'}, inplace=True)

    

       
#%%
def ToExcel():  
    
       
    for p in people_comp_dic:
        with pd.ExcelWriter('NewsFor' + str(p)+'-'+ DT.date.today().strftime("%d-%m-%Y") +'.xlsx',engine='xlsxwriter',options={'strings_to_urls': False}) as writer:
            workbook=writer.book
            cell_format = workbook.add_format()
            cell_format.set_text_wrap({'text_wrap':True})
                        
            col_format = workbook.add_format({ 
                                   'align': 'vcenter',                                   
                                   'text_wrap': 'vjustify',
                                   'num_format':'@'})
            #print(i)
                
            for c in range(len(people_comp_dic[p])) :
                
                if len(people_comp_dic[p][c]) == 2 and type(people_comp_dic[p][c][1]) == list: 
                    
                    GroupedDataset=pd.DataFrame()
                       #print(GroupedDataset)
                       

                    for sc in range(len(people_comp_dic[p][c][1])):
                        for i in general_df:
                            if str(i) == str(people_comp_dic[p][c][1][sc]):
                                GroupedDataset=pd.concat([GroupedDataset,general_df[i]],axis=0,sort=False)
                            else:
                                None
                    
                    #if GroupeDataset = []:
                    #print(GroupedDataset)  
                    if GroupedDataset.empty == False:
                        
                        GroupedDataset.to_excel(writer,sheet_name=str(people_comp_dic[p][c][0]), index=False)
                        worksheet=writer.sheets[str(people_comp_dic[p][c][0])]
                        worksheet.autofilter('A1:H20')
                        worksheet.set_column('A:A',10,col_format)
                        worksheet.set_column('B:B',10,col_format)
                        worksheet.set_column('C:C',10,col_format)
                        worksheet.set_column('D:D',40,col_format)
                        worksheet.set_column('E:E',30,col_format)
                        worksheet.set_column('F:F',40,col_format)
                        worksheet.set_column('G:G',10,col_format)
                        worksheet.set_column('H:H',30,col_format)
                        worksheet.set_column('I:I',30,col_format)
                        
                    else:
                        #print('df empty')
                        GroupedDataset.to_excel(writer,sheet_name=str(people_comp_dic[p][c][0]), index=False)
                        worksheet=writer.sheets[str(people_comp_dic[p][c][0])]
                
                        worksheet.write('A1', 'No news about this group.')
                    

                       
                       
                elif len(people_comp_dic[p][c]) > 1:
                    for sc in range(len(people_comp_dic[p][c])):
                        for i in general_df:
                            if str(i) == str(people_comp_dic[p][c][sc]):
                                general_df[i].to_excel(writer,sheet_name=str(i), index=False)
    
                                worksheet=writer.sheets[str(i)]
                                worksheet.autofilter('A1:H20')
                                worksheet.set_column('A:A',10,col_format)
                                worksheet.set_column('B:B',10,col_format)
                                worksheet.set_column('C:C',10,col_format)
                                worksheet.set_column('D:D',40,col_format)
                            #worksheet.set_column('D:D',20,col_format)
                                worksheet.set_column('E:E',30,col_format)
                                worksheet.set_column('F:F',40,col_format)
                                worksheet.set_column('G:G',10,col_format)
                                worksheet.set_column('H:H',30,col_format)
                                worksheet.set_column('I:I',30,col_format)
        
        
                            else:
                                None
                else:
                    None 
                    
            if RSSfeed.empty == True:
                RSSfeed.to_excel(writer,sheet_name='Gartner', index=False)
                worksheet=writer.sheets['Gartner']
                
                worksheet.write('A1', 'No Gartner feeds for today!')
            else:
                
                
            
                RSSfeed.to_excel(writer,sheet_name='Gartner',index=False)
                worksheet=writer.sheets['Gartner']
                worksheet.autofilter('A1:G20')
                worksheet.set_column('A:A',10,col_format)
                worksheet.set_column('B:B',40,col_format)
                worksheet.set_column('C:C',30,col_format)
                worksheet.set_column('D:D',30,col_format)
                worksheet.set_column('E:E',30,col_format)
                worksheet.set_column('F:F',40,col_format)
                worksheet.set_column('G:G',10,col_format)
                worksheet.set_column('H:H',30,col_format)
    
            
            CS.to_excel(writer,sheet_name='Info',index=False)
            
            worksheet=writer.sheets['Info']

            worksheet.set_column('A:B',30,col_format)
            
            GroupedDuplicate=pd.DataFrame()
            for d in range(len(duplicate_df)):
                GroupedDuplicate=pd.concat([GroupedDuplicate,duplicate_df[d]],sort=False,axis=0)
            
            for tw in range(len(tw_duplicate_df)):
                GroupedDuplicate=pd.concat([GroupedDuplicate,tw_duplicate_df[tw]],sort=False,axis=0)
            
            for db in range(len(Double_df)):
                #for i in range(len(Double_df[db])):
                GroupedDuplicate=pd.concat([GroupedDuplicate,Double_df[db]],sort=False,axis=0)
                    
            
            if GroupedDuplicate.empty == False:
                
                GroupedDuplicate.to_excel(writer,sheet_name='Backlog', index=False)
                worksheet=writer.sheets['Backlog']
                worksheet.autofilter('A1:H20')
                worksheet.set_column('A:A',10,col_format)
                worksheet.set_column('B:B',10,col_format)
                worksheet.set_column('C:C',10,col_format)
                worksheet.set_column('D:D',40,col_format)
                worksheet.set_column('E:E',30,col_format)
                worksheet.set_column('F:F',40,col_format)
                worksheet.set_column('G:G',10,col_format)
                worksheet.set_column('H:H',30,col_format)
            
            else:
                GroupedDuplicate.to_excel(writer,sheet_name='Backlog', index=False)
                worksheet=writer.sheets['Backlog']
                
                worksheet.write('A1', 'No duplicates.')              
            
        
    # ALL COMPANIES
    
                
    with pd.ExcelWriter('AllCompaniesNews.xlsx',engine='xlsxwriter') as writer:
        for i in general_df:
            general_df[i].to_excel(writer,sheet_name=str(i), index=False)
            worksheet=writer.sheets[str(i)]
            worksheet.set_column(0,0,20)
            worksheet.set_column(1,1,80)
            worksheet.set_column(2,2,30)
            worksheet.set_column(3,3,30)
            worksheet.set_column(4,4,30)
            worksheet.set_column(5,5,100)
            for row in range(len(general_df[i].index)):
                            worksheet.set_row(row,15,cell_format)

           # datasets[i].style.apply(format_df)

ToExcel() 
#%%
    #email 
def SendEmail():    
    outlook = win32.Dispatch('outlook.application')
    

    
    #replace test with people_email_dic
    
    for p in people_email_dic:
        #print(p)
        mail = outlook.CreateItem(0)
        mail.To = str(people_email_dic[p]) +';'
        mail.Subject = 'Berguig News for '+ str(p)
        mail.Body = 'Message body er56oijtyg7po'
        email_p1 = '<p>Dear '+str(p)+',</p><p>Please find attached last 24 hour news about '
        email_p2= ''
        email_p3=''

        for c in people_comp_dic:
            if c == p:
            
                for i in range(len(people_comp_dic[c])):
                    
                    if len(people_comp_dic[c][i]) == 2 and type(people_comp_dic[c][i][1]) == list:
                        email_p2 = email_p2 + str(people_comp_dic[c][i][0]) + ' group, '
                    
                    else:
                        
                        
                        for z in range(len(people_comp_dic[c][i])):
                            
                            email_p3 = email_p3 + str(people_comp_dic[c][i][z]) + ', '
            else:
                None
                
        email_p4='which you have subscribed to.</p><p>Do feel free to revert if you want to subscribe to any more company news.</p><p>Regards,</p><p>Genie</p>' #this field is optional
        
        mail.HTMLBody= email_p1 + email_p2 + email_p3 + email_p4
        
        attachment = 'C:\\Users\\path\\NewsFor'+str(p)+'-'+ DT.date.today().strftime("%d-%m-%Y") +'.xlsx'
        mail.Attachments.Add(attachment)
        mail.Send()


SendEmail()


#%%
import xml.etree.ElementTree as ET

def log_creator():
    
    today=DT.date.today()
    #today=today.strftime("%Y-%m-%d")
    
    root = ET.Element("date")
    #doc= ET.SubElement(root, "doc")
    
    ET.SubElement(root, "day").text = str(today)
    
    tree = ET.ElementTree(root)
    tree.write("log.xml")

log_creator()    

#%%
