from transformers import AutoModelForCausalLM, AutoTokenizer
from pandas import DataFrame, read_json 
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
import logging
from tqdm import tqdm

logging.basicConfig(
    filename = 'surprisal.log',
    level=logging.INFO,          # minimum level to log
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info("importing done, starting program")


MODEL_NAME = 'openai-community/gpt2-medium'
DATAFRAME = read_json('data/dev.json', orient = 'index')
DATAFRAME_LENGTH = DATAFRAME.__len__()
logging.info("Dataframe constructed")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval() # sets training mode to False


# sentence = "The potential (the inherent capacity for coming into being) couldn't be measured."

"""logits gives the params for the next predictions. so choose item_id of
next item, not the current one."""


logging.info("defining functions")

def average_surprisal(surprisals):
    result = np.sum(surprisals)/len(surprisals)
    return result


def linear_transformation(a,b):
    """
    - Tries to transform list 'a' into list 'b', with linear transforms.
    - Returns a list.
    """
    a = np.array(a)
    b = np.array(b)
    replacement_val = a[~np.isnan(a)].mean()
    a[np.isnan(a)] = 0
    print(f"gold_data mean: {b.mean()}")
    print(f"gold_data std: {b.std()}")
    print(f"pred_data mean: {a.mean()}")
    print(f"pred_data std: {a.std()}")
    print(f"nan values: {np.isnan(a)}")
    average_a = (a - a.mean())/a.std()
    mapping_to_b = average_a * b.std()
    result = mapping_to_b + b.mean()
    return  result


def calc_surprisals(sentence):
    """
    - calculates the surprisal of every token in the sentence
    - returns an np.array with all surprisals"""

    def calc_input_ids():
        """tokenization"""
        input = tokenizer(sentence, return_tensors = 'pt') #returns a pytorch tensor
        input_ids = input["input_ids"]
        return input_ids

    def calc_logits():
        """gets predictions scores of the modelling head"""
        with torch.no_grad():
            output = model(input_ids, labels = input_ids)
            logits = output.logits
        return logits

    input_ids = calc_input_ids()        
    logits = calc_logits()
    
    surprisals = np.empty(input_ids.size(1))
    log_probabilities = torch.log_softmax(logits, dim= -1)

    for i in range(1, input_ids.size(1)):
        id = input_ids[0, i]
        token_log_prob = log_probabilities[0, i-1, id].item() 
        surprisals[i - 1] = - token_log_prob 
        if token_log_prob is None:
            print(i)
    return surprisals
    





def combine_sentences_with_brackets (dataset): 
    """
    - combines the 'sentence' with the judged meaning of homonym
    - without 'ending'
    - returns a generator
    """

    def splitting_after_homonym(sentence, homonym_pattern):
        """splitting 'sentence' after the homonym"""
        word_list = sentence.split(' ')
        for index, current_word in enumerate(word_list):
            word = re.match(homonym_pattern, current_word)
            if word != None: 
                first_half = word_list[:index +1]
                second_half = word_list[index +1:]
                return ' '.join(first_half), ' '.join(second_half)
        return -1
        

    for i,series in dataset.iterrows():
        judged_meaning = series['judged_meaning']
        homonym = series['homonym']
        sentence= series['sentence']
        f =  splitting_after_homonym
        try:
            sent_before_h, sent_after_h = f(sentence, homonym)
        except TypeError as e:
            print(f'ERROR: Still working on regex for homonym! row {i}')
            continue

        combined_sentence = sent_before_h + ' (' + judged_meaning + ') ' + sent_after_h
        yield combined_sentence

logging.info("function definitions ready")




logging.info("Sentence computation starting")

g = combine_sentences_with_brackets(DATAFRAME)
# sents = [next(g) for i in range(300)]
sents = list(g)
results = []

with tqdm(total= 100) as pbar:
    for position, sentence in enumerate(sents):

        surprisals = calc_surprisals(sentence)
        result = average_surprisal(surprisals)
        results.append(result) 
        
        pbar.update(np.ma.round(100/DATAFRAME_LENGTH, 2))
    

gold_data = list( DATAFRAME['average'] )
final_values = linear_transformation(results, gold_data)

logging.info("linear transformation succeded")

logging.info("making Datafram")

final_values = np.ma.round(final_values, decimals= 0)
index = np.arange(len(final_values)) 
res = DataFrame({'index': index, 'predictions': final_values})

logging.info("dumping Datafram")

res.to_json("results/predictions.jsonl", orient= 'records', lines= True)

logging.info("done")