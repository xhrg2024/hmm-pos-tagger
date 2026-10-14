# PKU-SEGPOS

### Introduction
PKU_SEGPOS is the part-of-speech (POS) tagging dataset provided by Peking University to tag the corpus of People's Daily. The corpus from January 2000 and December 1-15, 2020, the corpus from December 16-23, 2000, and the corpus from December 24-31, 2000 are the training set, the validation set, and the test set, respectively. 

The POS tagging uses 26 POS tags (noun n, time word t, location word s, location word f, numeral m, quantifier q, distinguishing word b, pronoun r, verb v, Adjective a, state word z, adverb d, preposition p, conjunction c, auxiliary word u, modal particle y, interjection e, onomatopoeia o, idiom i, idiom l, abbreviation j, preceded component h, followed component k, morpheme g, non-morpheme character x, punctuation mark w) in the "Modern Chinese Grammar Information Dictionary", and add the following three types of marks: ①The classification mark of proper nouns, namely person name nr, place name ns, group organization unit name nt, other proper nouns nz; ②Subcategory markers of morphemes, namely noun morpheme Ng, verb morpheme Vg, adjective morpheme Ag, tense morpheme Tg, adverb morpheme Dg, etc.; ③Subcategory markers of verbs and adjectives, namely noun verb vn (verbs with noun characteristics), Nouns an (adjectives with noun characteristics), adverbs vd (verbs with adverb characteristics), adverbs ad (adjectives with adverb characteristics). There are about 40 in total.


### Data Size
Training set: 31,772; Validation set: 5,204; Test set: 4,764.

### Data Format
Each instance is composed of text(tokens, a list) and POS information of each token in the text (span_list, a list), where each POS information contains the starting position, ending position and category information.

### Example
```
{"tokens": ["一", "种", "激", "励", "机", "制"], "span_list": [{"start": 0, "end": 0, "type": "m"}, {"start": 1, "end": 1, "type": "q"}, {"start": 2, "end": 3, "type": "vn"}, {"start": 4, "end": 5, "type": "n"}]}
```

### Evaluation Code
The prediction result needs to be in the same format as the evaluation code.
The correct file name is PKUSEGPOS.jsonline
```shell
python eval.py prediction_file test_private_file
```
The evaluation metrics are P, R, F1, and the output is in dictionary format.
```she
return {"P":_, "R":_, "F1":_}
```

### Author List
Huiming Duan, Baobao Chang, Zhifang Sui

### Institutions
Peking University


