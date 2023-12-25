from constant import OPENAI_API_KEY
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
llm=OpenAI(temperature=0.7)
st.title("  - :  Common Weather of cities : -  ")

language=st.sidebar.selectbox(
    label="select any Language",
    options=("English", "Mandarin Chinese", "Hindi", "Spanish", "French", 
             "Modern Standard Arabic", "Bengali", "Portuguese", "Urdu", 
             "Indonesian", "German", "Italian", "Turkish", "Vietnamese", 
             "Russian", "Thai", "Tamil", "Yue Chinese", "Marathi", 
             "Telugu", "Japanese", "Western Punjabi", "Wu Chinese", 
             "Korean", "French Creole", "Cantonese", "Malay", "Telugu", 
             "Urdu", "Gujarati", "Javanese", "Southern Pashto", "Burmese", 
             "Hakka Chinese", "Tagalog", "Ukrainian", "Yoruba", "Maithili", 
             "Uzbek", "Sindhi", "Amharic", "Farsi", "Yoruba", "Malayalam", 
             "Igbo", "Sundanese", "Dutch", "Kurdish", "Thai", "Egyptian Arabic", 
             "Filipino", "Kannada", "Moroccan Arabic", "Hausa", "Burmese", 
             "Polish", "Serbo-Croatian", "Nepali", "Sinhalese", "Kirundi", 
             "Zulu", "Czech", "Kinyarwanda", "Uyghur", "Swedish", 
             "Haitian Creole", "East Javanese", "Finnish", "Bhojpuri", 
             "Oromo", "Bulgarian", "Fula", "Malay", "Bambara", "Ilokano",
               "Hejazi Arabic", "Igbo", "Dinka", "Somali", "Latvian", 
               "Tajik", "Lithuanian", "Bashkir", "Kazakh", "Lao", "Lingala", 
               "Tatar", "Tswana", "Bavarian", "Low German", "Akan", "Aragonese",
                 "Batak Toba", "Bavarian", "Low German"),
    )
mood=str(st.sidebar.multiselect(
    label="select output text style :-",
    options=("Funny", "Happy", "Excited", "Anxious",
             "Angry", "Content", "Frustrated", 
             "Confident", "Nervous", "Relaxed", 
             "Motivated", "Bored", "Surprised", 
             "Amused", "Enthusiastic", "Disappointed", 
             "Curious", "Pensive", "Grateful", "Hopeful"
)
))

city=st.text_input("Enter City Name",None)
country=st.text_input("Enter Country Name",None)

first_prompt=PromptTemplate(
    input_variables=["language","mood","city","country"],
    template="""What is the weather of {city} city of {country}.
    Considered your writing mood is {mood},
    and answer should be in {language} language"""
)

first_memory=ConversationBufferMemory(
    input_key="city",
    memory_key="result"
)

first_chain=LLMChain(
    llm=llm,
    prompt=first_prompt,
    verbose=True,
    memory=first_memory
)
if language and mood and city and country:
    output=first_chain(
        {"language":language,"mood":mood,"city":city,"country":country})
    st.write(output)
    with st.expander('History: '):
        st.info(first_memory.buffer)