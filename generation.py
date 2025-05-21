import pandas as pd
import logging
from langchain_core.messages import HumanMessage
from helpers import (get_llm)



logger = logging.getLogger(__name__)

def abs_generation(df,output_path,llm_type,llm_model):
    print("abs generation")
    llm=get_llm(llm_type,llm_model)
    claim=df['claim']
    df['generated_abstract'] = ''  
    def generate_abstract(claim):
        prompt = f'''You are a patent expert.Given the following patent claim, write an informative abstract that captures  the key invention, technical purpose, functionality. Do not write title or anything else.

                    Patent Claim: {claim}

                    Abstract:'''
                            
        msg = llm.invoke([HumanMessage(content=prompt)])
        print(msg.content)
        return msg.content
        

    df.iloc[0:0].to_csv(output_path, index=False)
    
    for index, row in df.iterrows():
            print(index)
            generated_abstract = generate_abstract(row['claim'])
            df.at[index, 'generated_abstract'] = generated_abstract if isinstance(generated_abstract, str) else 'Error or empty'
            df.iloc[[df.index.get_loc(index)]].to_csv(output_path, mode='a', header=False, index=False)

def claim_generation(df,output_path,llm_type,llm_model):
    print("abs generation")
    llm=get_llm(llm_type,llm_model)
    claim=df['abstract']
    df['generated_claim'] = ''  
    def generate_claim(abstract):
        prompt = f'''this is a patent abstract:{abstract}. Generate the claim for this abstract'''
        msg = llm.invoke([HumanMessage(content=prompt)])
        print(msg.content)
        return msg.content


    df.iloc[0:0].to_csv(output_path, index=False)
    
    for index, row in df.iterrows():
            print(index)
            generated_abstract = generate_claim(row['abstract'])
            df.at[index, 'generated_claim'] = generated_abstract if isinstance(generated_abstract, str) else 'Error or empty'
            df.iloc[[df.index.get_loc(index)]].to_csv(output_path, mode='a', header=False, index=False)
