### Embddings

#### 1.
pretext satz und ending mit meaning yu einem embedding
--> ca 55%
- something with regression
- something more with rounding the values 

#### 2.
pretext-satz-ending enbedding
similarity mit meaning berechnen
 
## possible models
- s-bert
- e5

Paper with model for model finding


### todo

- '(' ')' vs '-'
- period in middle of sentence
- literatur

instead of similarity:
- classification
- embedding with presentence sentence and ending
- find correct meaning (mostly biniary)
- - give chance for training something


#### Homework

- [ ] some kind of grafics for easily understanding 
the problems of a project
- [ ] slides maybe
- [ ] evaluations ***script*** aus dem github


Standard deviation how np does it:
 $$\sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}
$$ 
it does std on the average
for some reason the ddof value of standart deveation 
seems to be 1. why do they use it??

Surprisal:
- good for singular tokens

$Surprisal(w_{i+1})= -log_2(p(w_{i+1}|w_i,...,w_0))$

- since it is only for singular tokens, 
take average surprisal for whole sentence
- [ ? ] take surprisal only for ending
- [ ? ] surprisal only for judged meaning

Perplexity:
- good sequences of tokens

$PP(p) = \prod_x p(x)^{-p(x)} = 
    b^{-\sum_x p(x) log_b (p(x))}$

You can interpret the exponent 
$-\sum_x p(x) log_b (p(x))$ as the cross-entropie 
$H(p,q) = -\sum_x p(x) log_b (q(x))$


Since you want to compare how surprised the model is 
by sentences, see if perplexity gives a better result,
than surprisal.



GPT-2 Model:
use  `'GPT2 specific outputs`--> 

`logits (torch.FloatTensor of shape (batch_size, num_choices, sequence_length, config.vocab_size))` — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).


from `PreTrainedTokenizer` class in `__call__` 
using `return_tensors = 'pt'` for pytorch tensors


we get something like `torch.Size([1, 6, 50257])`

`torch.no_grad()` can help to reduce storage - no gradients stored/calculated

#### Problems

- [ ] fixing nan problem better
- [ ] wrong surprisal numbers