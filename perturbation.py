import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')


columns = ['claim', 'typo_claim', 'synonym_claim', 'bert_claim', 'swapped_claim', 'combined_claim']
df = pd.read_csv("data/A61.csv")
df=df[:1000]

for col in columns[1:]:
    df[col] = ""

output = "A61_perturbation_1000.csv"
df.iloc[0:0][columns].to_csv(output, index=False) 

typograph= nac.KeyboardAug()
syn = naw.SynonymAug(aug_src='wordnet')
bertcont = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
swapped = naw.RandomWordAug(action="swap")

for idx, row in df.iterrows():
    text = row['claim']
    if isinstance(text, str):
        try:
            typo = typograph.augment(text)
            synonym = syn.augment(text)
            bert = bertcont.augment(text)
            swap = swapped.augment(text)

            combined = swapped.augment(
                          bertcont.augment(
                            syn.augment(
                              typograph.augment(text)
                            )
                          )
                      )

            new = pd.DataFrame([{
                'claim': text,
                'typo_claim': typo,
                'synonym_claim': synonym,
                'bert_claim': bert,
                'swapped_claim': swap,
                'combined_claim': combined
            }])
            print(idx)

            new.to_csv(output, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error at row {idx}: {e}")
