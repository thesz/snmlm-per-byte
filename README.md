# Sparse non-negative language model, byte level

Sparse non-negative language modeling (SNMLM): https://arxiv.org/abs/1412.1454

This little "project" of mine implements it with twists.

# The Twists

First, the online learning. It appears that just gradient descent is good enough.

Second, SNMLM uses counts to give "ground truth" estimation, this thing does not, it bets all to the mixture of exponentials.

Third, it uses an entropy as a part of feature vector. And feature vector is very short: feature hash, feature+target hash and entropy multiplier.

Fourth, estimation goes not deeper than first context with just one character to predict. This is taken from PPM* algorithm.

And, last but not least, no skip-grams. For now.

There are tables for context bitmasks and for weights. The targets for some context's hash are recorded in the masks. So, probably, it is also a twist: features can collide not only on weights, but on set of possible targets.

# Performance

It's prediction of next byte averages on enwik8 (see Hutter Prize) to 1.653 bits per byte. This is just 0.01 shy of LSTM implementation of lstm-compress and Bellard's small LSTM in his NNCP project.

NNCP: https://bellard.org/nncp/nncp.pdf

The speed is very much adequate, about 30Kbytes/sec on my i7-10875H CPU @ 2.30GHz.

# Why?

I don't know, mostly out of curiosity. Everyone went the same way, it seems lately, and going a different direction can be fruitful. Or fun.

Also I remembered that my implementation of SNMLM was also comparable to LSTM LM during my stint in Yandex. Looks like my memory is not incorrect.
