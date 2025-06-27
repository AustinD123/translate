import numpy as np
from collections import Counter

class BPE():
    def __init__(self, text,num_merges):
        self.num_merges=num_merges
        self.text = text
        self.vocab = {}
        self.counts = {}
        self.token2id = {}
        self.id2token = {}
        self.next_id = 256  

    def tokens(self):
        tokens = list(self.text.encode("utf-8"))
        return tokens

    def get_stats(self, tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens, pair, idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def merge(self, stats, tokens):
        if not stats:
            return tokens
        top = max(stats, key=stats.get)
        new_id = self.next_id
        self.next_id += 1
        self.token2id[top] = new_id
        self.id2token[new_id] = top
        tokens = self.merge_tokens(tokens, top, new_id)
        return tokens

    def run(self):
        tokens = self.tokens()
        for _ in range(self.num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            tokens = self.merge(stats, tokens)
        return tokens

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break
            top = max(stats, key=stats.get)
            if top not in self.token2id:
                new_id = self.next_id
                self.token2id[top] = new_id
                self.id2token[new_id] = top
                self.next_id += 1
            else:
                new_id = self.token2id[top]
            tokens = self.merge_tokens(tokens, top, new_id)
        return tokens

    def decode(self, tokens):
        def decode_token(t):
            if t < 256:
                return [t]
            else:
                pair = self.id2token[t]
                return decode_token(pair[0]) + decode_token(pair[1])
        
        byte_list = []
        for t in tokens:
            byte_list.extend(decode_token(t))
        return bytes(byte_list).decode("utf-8", errors="replace")
