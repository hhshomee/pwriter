import pandas as pd
import sys
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import argparse

sys.setrecursionlimit(5000)

def run_eval(input,output,part):
    df = pd.read_csv(input)
    print(df.columns)
    
    if part=='abs':
        candidates = df['generated_abstract'].astype(str).tolist()
        references = df['abstract'].astype(str).tolist()
    elif part=='claim':

        candidates = df['generated_claim'].astype(str).tolist()
        references = df['claim'].astype(str).tolist()


    print("Calculating BERTScore")
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    df['bert_Precision'] = [float(p) for p in P]
    df['bert_Recall'] = [float(r) for r in R]
    df['bert_F1_Score'] = [float(f) for f in F1]


    print("Calculating BLEU score")
    def bleu_scores(refs, gens):
        scores = []
        for ref, gen in zip(refs, gens):
            reference_tokens = ref.split()
            generated_tokens = gen.split()
            cc = SmoothingFunction()
            score_val = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=cc.method1)
            scores.append(score_val)
        return scores

    bleu_scores = bleu_scores(references, candidates)
    df['bleu_score'] = [float(f"{score:.2f}") for score in bleu_scores]

    print("Calculating Cosine Similarity")
    cosine_scores = []
    for ref, gen in zip(references, candidates):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([ref, gen])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        cosine_scores.append(cosine_sim)
    df['cosine_similarity'] = [float(f"{score:.2f}") for score in cosine_scores]


    print("Calculating ROUGE score")
    rouge = Rouge()
    rouge_precision, rouge_recall, rouge_f1 = [], [], []

    for ref, gen in zip(references, candidates):
        try:
            scores = rouge.get_scores(gen, ref)[0]['rouge-l']
            rouge_precision.append(scores['p'])
            rouge_recall.append(scores['r'])
            rouge_f1.append(scores['f'])
        except:
            rouge_precision.append(0.0)
            rouge_recall.append(0.0)
            rouge_f1.append(0.0)

    df['rouge_Precision'] = [float(f"{p:.2f}") for p in rouge_precision]
    df['rouge_Recall'] = [float(f"{r:.2f}") for r in rouge_recall]
    df['rouge_F1_Score'] = [float(f"{f:.2f}") for f in rouge_f1]

    df.to_csv(output, index=False)
    print(f"All metrics saved to: {output}")

if __name__ =="__main__":
    parser=argparse.ArgumentParser(description="Run generation and evaluation")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output_prefix", "-o",  help="Prefix for output file")
    parser.add_argument("--part", type=str, help="abs/claim which one you want to generate")
    args = parser.parse_args()

    run_eval(args.input, args.output_prefix, args.part)

#python evaluation.py --input results/A61_GPT4_abs.csv --output_prefix results/evaluation/eval_A61_GPT4_abs.csv --part abs 

