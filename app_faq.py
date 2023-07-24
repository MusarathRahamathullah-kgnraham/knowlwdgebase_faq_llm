
import os
import time
import ai21
import json
import boto3
import pandas as pd
import streamlit as st

from PIL import Image
from io import BytesIO
from collections import deque
from datetime import datetime
#from pdf2image import convert_from_bytes
st.set_page_config(layout="wide")

bedrock = boto3.client('bedrock', region_name='us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com')

APP_MD    = json.load(open('application_metadata.json', 'r'))
MODELS    = {d['name']: d['endpoint'] for d in APP_MD['models']}
#MODEL_SUM = APP_MD['summary_model']
REGION    = APP_MD['region']
BUCKET    = APP_MD['datastore']['bucket']
PREFIX    = APP_MD['datastore']['prefix']
KENDRA_ID = APP_MD['kendra']['index_id']
#CONTEXT = deque([''], maxlen=10)

S3            = boto3.client('s3', region_name=REGION)
TEXTRACT      = boto3.client('textract', region_name=REGION)
KENDRA        = boto3.client('kendra', region_name=REGION)
SAGEMAKER     = boto3.client('sagemaker-runtime', region_name=REGION)

CHAT_FILENAME = 'chat.csv'
params = {'file':'','action_name':'','endpoint':'', 'max_len':0, 'top_p':0, 'temp':0, 'model_name':''}


def query_endpoint(endpoint_name, prompt_data,params):
    accept='application/json'
    contentType='application/json'
    
    # if 'huggingface' in endpoint_name:
    #     response = SAGEMAKER.invoke_endpoint(
    #         EndpointName=endpoint_name,
    #         ContentType=contentType,
    #         Body=json.dumps(payload).encode('utf-8')
    #     )
    #     output_answer = json.loads(response['Body'].read().decode('utf-8'))["generated_texts"][0]
    if 'claude' in endpoint_name:
        body=json.dumps({"prompt": prompt_data, "max_tokens_to_sample":params['max_len'],"temperature":params['temp'],"top_p":params['top_p']})
        response = bedrock.invoke_model(
            body=body, 
            modelId='anthropic.claude-instant-v1', 
            accept=accept, 
            contentType=contentType
        )
        #print(response)
        output_answer = json.loads(response.get('body').read()).get('completion')
    elif 'titan' in endpoint_name:
        #print(json.dumps(prompt_data))
        body=json.dumps({"inputText": prompt_data,"textGenerationConfig": {
                          "maxTokenCount": params['max_len'],
                          "temperature":params['temp'],
                          "topP":params['top_p']
                         }})
        response = bedrock.invoke_model(
            body=body, 
            modelId='amazon.titan-tg1-large', 
            accept=accept, 
            contentType=contentType
        )
        output_answer = json.loads(response.get('body').read()).get('results')[0].get('outputText')
    elif 'j2' in endpoint_name:
        body = json.dumps({"prompt": prompt_data,"maxTokens":params['max_len'],"temperature":params['temp'],"topP":params['top_p']})
        print(json.dumps(prompt_data))
        response = bedrock.invoke_model(
            body=body, 
            modelId= 'ai21.j2-grande-instruct', # change this to use a different version from the model provider, 
            accept=accept, 
            contentType=contentType)
        output_answer = json.loads(response.get('body').read()).get('completions')[0].get('data').get('text')
    
    #print(output_answer)
    return str(output_answer)

def query_index(query):
    response = KENDRA.query(
        QueryText = query,
        IndexId = KENDRA_ID
        
    )
    return response


def extract_text(bucket, filepath):
    response = TEXTRACT.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket':bucket, 'Name':filepath}})
    text = TEXTRACT.get_document_text_detection(JobId=response['JobId'])
    i = 0
    while text['JobStatus'] != 'SUCCEEDED':
        time.sleep(5)
        i += 1
        text = TEXTRACT.get_document_text_detection(JobId=response['JobId'])
        if i >= 10:
            text = ''
            break
    text = '\n'.join([t['Text'] for t in text['Blocks'] if t['BlockType']=='LINE'])
    return text


def load_document(file_bytes):
    # try:
    #     images = convert_from_bytes(file_bytes)
    #     image_page_1 = images[0].convert('RGB')
    #     st.image(image_page_1)
    # except:
    #     st.write('Cannot display image. Ensure that you have poppler-utils installed.')
    
    with open('doc.pdf', 'wb') as fp:
        fp.write(file_bytes)
    with open('doc.pdf', 'rb') as fp:
        S3.upload_fileobj(fp, BUCKET, PREFIX+'/doc.pdf')
    time.sleep(2)
    text = extract_text(BUCKET, PREFIX+'/doc.pdf')
    return text


def summarize_context(context, params):
    try:
        prompt_data ="""'"""+context+"""\n"""+"summarize the context"+"""'"""
        output_summary = query_endpoint(params['endpoint'],prompt_data, params)
        return output_summary
            
    except:
        return 'No summarization endpoint connected'

def action_qna(params):
    st.title('Ask Questions of your Model')
    try:
        chat_df = pd.read_csv(CHAT_FILENAME)
        
    except:
        chat_df = pd.DataFrame([], columns=['timestamp', 'question', 'response'])
    kendra_links = []
    
    input_question = st.text_input('**Please ask a question:**', '')
    if st.button('Send Question') and len(input_question) > 3:
        response = query_index(input_question)
        #print("response:",response['ResultItems'])
        for sr in response['ResultItems']:
            # kendra_links.append(sr['DocumentURI'])
            if sr['ScoreAttributes']['ScoreConfidence'] == 'HIGH':
                kendra_links.append(sr['DocumentURI'])
                # st.write(f"[Link to Source Document]({sr['DocumentURI']})")
                # st.write(f"**[{sr['ScoreAttributes']['ScoreConfidence']}]** | {sr['DocumentTitle']['Text']} [Link to Source Document]({sr['DocumentURI']})")
                # st.write(sr['DocumentExcerpt']['Text'])
                # # st.write('---')
        print("Kendra Links:",kendra_links)        
        # kendra_links = list(set(kendra_links))
        kendra_context = '\n'.join([sr['DocumentTitle']['Text']  for sr in response['ResultItems'] if sr['ScoreAttributes']['ScoreConfidence'] == 'HIGH'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        # context = '\n'.join(['Context: ' + str(r.question) + '\nResponse: ' + str(r.response) + '\n' for idx,r in chat_df.iloc[-1:].iterrows()])
        # context = context + kendra_context
        print("kendra context:",kendra_context)
        #payload = {
        #     "text_inputs": context + '\n' + input_question, #input_question,
        #     "max_length": params['max_len'],
        #     "max_time": 50,
        #     "num_return_sequences": 1,
        #     "top_k": 50,
        #     "top_p": params['top_p'],
        #     "do_sample": True,
        # }
        prompt_data ="""'"""+'Read the following passage and answer the questions that follow:\n'+kendra_context+"""\n"""+'Questions:'+input_question+"""\n"""+'Answer:'+"""'"""


        output_answer = query_endpoint(params['endpoint'], prompt_data, params)
        st.text_area('Response:', output_answer)
        for each_link in kendra_links[0:1]:   #        
            st.write(f"[Link to Source Document]({each_link})") 
        chat_df.loc[len(chat_df.index)] = [timestamp, input_question, output_answer]
        chat_df.tail(5).to_csv(CHAT_FILENAME, index=False)
                    
    st.subheader('Recent Questions:')
    for idx,row in chat_df.iloc[::-1].head(5).iterrows():
        st.write(f'**{row.timestamp}**')
        st.write(row.question)
        st.write(row.response)
        st.write('---')


# def action_search(params):
#     st.title('Ask Questions of your Document')
#     #col2 = st.columns(1)
#     #with col2:
#     input_question = st.text_input('**Please ask a question of a loaded document:**', '')
#     if st.button('Send Question') and len(input_question) > 3:
#         # LLM 
#         payload = {
#             "text_inputs": input_question,
#             #"max_length": params['max_len'],
#             "max_time": 50,
#             #"maxTokens": params['max_len'],
#             "num_return_sequences": 1,
#             "top_k": 50,
#             "temperature":params['temp'],
#             "top_p": params['top_p'],
#             "do_sample": True,
#         }
#         if params["model_name"] == "Bedrock Titan Model":
#                 output_answer = query_bedrock_endpoint(payload)

#         if "FLAN" in params["model_name"]:
#             #del payload['maxTokens']
#             payload['max_length'] = params['max_len']
#             output_answer = query_endpoint(params['endpoint'], payload)


#         elif "Jumbo" in params["model_name"]:
#             #del payload['max_length']
#             payload['maxTokens'] = params['max_len']
#             output_answer = query_endpoint(params['endpoint'], payload)
#         st.text_area('Response:', output_answer,height = 400)


def action_doc(params):
    st.title('Ask Questions of your Document')
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader('Upload a PDF file', type=['pdf'])
        if file is not None:
            context = load_document(file.read())
            if st.button('Summarize'):
                st.write('**Summary:**')
                st.write(summarize_context(context, params))
    with col2:
        input_question = st.text_input('**Please ask a question of a loaded document:**', '')
        if st.button('Send Question') and len(input_question) > 3:
            prompt_data ="""'"""+'Read the following passage and answer the questions that follow:\n'+context+"""\n"""+'Questions:'+input_question+"""\n"""+'Answer:'+"""'"""
            output_answer = query_endpoint(params['endpoint'],prompt_data, params)
            st.text_area('Response:', output_answer)


def app_sidebar():
    with st.sidebar:
        st.write('## How to use:')
        description = """Welcome to our LLM tool extraction and query answering application. With this app, you can aske general question, 
        ask questions of a specific document, or intelligently search an internal document corpus. By selection the action you would like to perform,
         you can ask general questions, or questions of your document. Additionally, you can select the model you use, to perform real-world tests to determine model strengths and weakneses."""
        st.write(description)
        st.write('---')
        st.write('### User Preference')
        if st.button('Clear Context'):
            pd.DataFrame([], columns=['timestamp', 'question', 'response']).to_csv(CHAT_FILENAME, index=False)
        action_name = st.selectbox('Choose Activity', options=['Question/Answer', 'Document Query' ]) #'Corpus Search',
        # if action_name == 'Corpus Search':
        #     while file is not None:
        #         file = st.file_uploader('Upload a PDF file', type=['pdf'])
        model_name = st.selectbox('Select Model', options=MODELS.keys())
        max_len = st.slider('Max Length', min_value=50, max_value=1500, value=150, step=10)
        top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
        temp = st.slider('Temperature', min_value=0.01, max_value=1., value=1., step=.01)
        st.write('---')
        st.write('## FAQ')
        st.write(f'**1. Where is the model stored?** \n\nThe current model is: `{model_name}` and is running within your account.')
        st.write(f'**2. Where is my data stored?**\n\n. Currently the queries you make to the endpoint are not stored, but you can enaable this by capturing data from your endpoint.')
        st.write('---')
        params['action_name']=action_name
        params['endpoint']=MODELS[model_name]
        params['max_len']=max_len
        params['top_p']=top_p
        params['temp']=temp
        params['model_name']=model_name
       
        # params = {'file':'','action_name':action_name,'endpoint':MODELS[model_name], 'max_len':max_len, 'top_p':top_p, 'temp':temp, 'model_name':model_name}
        return params


def main():
    params = app_sidebar()

    endpoint=params['endpoint']
    # if params['action_name'] == 'Corpus Search':
    #     params = action_search(params)
    if params['action_name'] == 'Question/Answer':
        params = action_qna(params)
    elif params['action_name'] == 'Document Query':
        params = action_doc(params)
    else:
        raise ValueError('Invalid action name.')


if __name__ == '__main__':
    main()

