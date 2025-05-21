import argparse
import pandas as pd
from data_loader import load_data
from generation import abs_generation,claim_generation

def run(input, output, limit,part, type, model):

    path=f"{output}.csv"
    df=load_data(input,limit)
    if part=='abs':
       
        print(df.head(5))


        abs_generation(
            df=df,
            output_path=path,
            llm_type=type,
            llm_model=model
        )
        print(f"Saved to {path}")
    elif part=='claim':
        
        print("head",df.head(5))
    
        claim_generation(
            df=df,
            output_path=path,
            llm_type=type,
            llm_model=model
        )
    print(f"Saved to {path}")


if __name__ =="__main__":
    parser=argparse.ArgumentParser(description="Run generation")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output_prefix", "-o",  help="Prefix for output file")
    parser.add_argument("--limit", "-l", type=int, help="Number of rows to process")
    parser.add_argument("--part", type=str, help="abs/claim which one you want to generate")
    parser.add_argument("--llm_type", type=str, default="llama", help="LLM type: openai, llama,deepseek")
    parser.add_argument("--llm_model", type=str, default="llama3", help="LLM model name like llama3.1 gpt-4 and so on...")
    args = parser.parse_args()

    run(args.input, args.output_prefix, args.limit,args.part, args.llm_type, args.llm_model)
    
#python main.py --input data/A61.csv --output_prefix results/A61_GPT4.1_abstracts --limit 8000  --part abs --llm_type openai --llm_model gpt-4.1