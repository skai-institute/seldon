import lightning.pytorch as pl
import os
import torch
import re
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import h5py
import glob
import random
from scipy.interpolate import interp1d
import json
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict


class JSONDataLoader(pl.LightningDataModule):

    def __init__(
        self,
        filename: str,
        context_length: int = 5,
        batch_size: int = 20,
        cache: bool = False,
        shuffle_training: bool = True,
        num_workers: int = 0,
    ):

        super().__init__()

        self.filename = filename

        self.cl = context_length

        self.batch_size = batch_size

        self.word2idx = {}
        self.idx2word = {}

        self.inputs = []
        self.outputs = []

        self.input_pad_mask_n = []
        self.output_pad_mask_n = []

        self.vocab = None
        self.merges = None
        self.first = True

        self.will_cache = cache
        self.is_cached = False
        self.cache = []

        self.sample = "train"

    def prepare_data(self):

        self.data = {}
        self.load_data()
        words = set()
        for data_source in self.json_data:
            self.data[data_source] = {}
            for i, obj in enumerate(self.json_data[data_source]):
                data = obj["lightcurve"]
                this_data = self.data[data_source]
                this_data[i] = {}
                this_data = this_data[i]

                this_data
                if "flux" not in data:
                    # print(data.keys())
                    continue
                if "obj_type" not in obj:
                    # print(obj.keys())
                    continue
                this_data["location"] = [data_source, obj["obj_type"], obj["object_id"]]
                this_data["band"] = data["band"]
                this_data["flux"] = np.array(data["flux"])
                this_data["flux"] /= this_data["flux"].max()
                this_data["flux"] = list(this_data["flux"])
                this_data["time"] = data["time"]

                N = len(this_data["band"])
                if N < self.cl:
                    continue
                low = N // 2 - N // 5
                high = N // 2 + N // 5
                if self.sample == "train":
                    this_data["band"] = this_data["band"][low:high]
                    this_data["flux"] = this_data["flux"][low:high]
                    this_data["time"] = this_data["time"][low:high]
                elif self.sample == "val":
                    this_data["band"] = (
                        this_data["band"][:low] + this_data["band"][high:]
                    )
                    this_data["flux"] = np.array(
                        this_data["flux"][:low] + this_data["flux"][high:]
                    )
                    this_data["time"] = np.array(
                        this_data["time"][:low] + this_data["time"][high:]
                    )

                # print(len(this_data['band']))
                words |= set(this_data["location"]) | set(data["band"])

        self.merges, self.vocab = bpe_encoding(list(words))
        words = self.vocab
        self.word2idx = dict(zip(words, range(len(words))))
        self.idx2word = dict(zip(range(len(words)), words))
        # if self.interpolate:
        #    self.init_interpolator()

    def load_data(self):
        with open(self.filename, "r") as f:
            self.json_data = json.load(f)

    def tokenize(self, data):

        input_tokens = tokenize(data, self.merges, self.vocab)
        return np.array(
            [self.word2idx[token] for token in input_tokens], dtype=np.int64
        )

    def pad_data(self, item, tokenize=False):
        N_location = len(item["location"])
        N_data = len(item["band"])
        N = N_location + N_data
        if N <= self.cl:
            # print(self.sample, 'BAD VALUES', N, self.cl)
            pass
        else:
            for i in range(0, N_data - self.cl, self.cl // 2):
                bands = item["band"][i : self.cl + i - N_location]
                mags = item["flux"][i : self.cl + i - N_location]
                times = item["time"][i : self.cl + i - N_location]
                inputs = ["<SOL>", *item["location"], *bands, "<EOL>"]
                if tokenize:
                    inputs = self.tokenize(inputs)
                    inputs = np.hstack(
                        (
                            inputs,
                            [self.word2idx["<PAD>"]] * (self.cl + 30 - len(inputs)),
                        )
                    )
                output_values = np.array([mags, times]).T
                # print(len(mags))
                yield inputs, output_values

    def iterate(self):

        for data_source in self.data:
            if data_source != "YSE":
                continue
            for obj in self.data[data_source]:
                # Want to sample this in some way (or sequence)
                # to have [*location, *bands], [times], [mags]
                item = self.data[data_source][obj]
                yield from self.pad_data(item, tokenize=True)

    def __iter__(self):

        return iter(self.iterate())

    def train_dataloader(self):
        self.sample = "train"
        self.prepare_data()

        dataset = DataLoader(
            list(self),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=self.shuffle_training,
            drop_last=True,
        )
        for i in dataset:
            len(i)
        dataset.num_workers = 0
        return dataset

    def val_dataloader(self):

        self.sample = "val"
        self.prepare_data()
        dataset = DataLoader(
            list(self),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )
        for i in dataset:
            len(i)
        dataset.num_workers = 0
        return dataset

    def test_dataloader(self):
        self.sample = "val"
        self.prepare_data()
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            num_workers=0,
        )

    def predict_dataloader(self):
        self.sample = "val"
        self.prepare_data()
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def get_vocab(data):
    """
    Given a list of strings, returns a dictionary of words mapping to their frequency
    count in the data.
    """
    # Considering adding a hook here for some random text for helping train the model
    # as well as providing some context for the tokenization selection
    vocab = defaultdict(int)
    for line in data:
        # vocab[' '.join(list(line)) + ' </endl>'] += 1
        # line = ' '.join(line)
        for word in line:
            # vocab[' '.join(list(word)) + ' </w>'] += 1
            vocab["".join(list(word))] = 2

    print("Made the base vocabulary")

    return vocab


def tokenize(text, merges, vocab):
    """text is already split or pre-tokenized"""
    # print('TEXT:', text)
    splits = [[*word] if word not in vocab else [word] for word in text]
    # print('SPLITS:', splits)
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1

            splits[idx] = split
    return sum(splits, [])


def bpe_encoding(data, vocab_size=1024):

    word_freqs = get_vocab(data)
    alphabet = get_alphabet(word_freqs)
    alphabet.sort()
    vocab = alphabet  # +should append the specials as well
    splits = {word: [*word] if word not in vocab else [word] for word in word_freqs}
    merges = {}
    bar = tqdm(total=vocab_size)
    bar.update(len(vocab))
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        if not pair_freqs:
            print(
                f"WARNING: Vocab is smaller than requested: {len(vocab)}/{vocab_size}"
            )
            break
        best = get_best_pair(pair_freqs)
        splits = merge_pair(*best, splits, word_freqs)
        merges[best] = "".join(best)
        vocab.append(merges[best])
        bar.update(1)
    bar.close()
    return merges, vocab


def get_best_pair(pair_freqs):

    return max(
        pair_freqs,
        key=lambda x: pair_freqs.get(x)
        - abs((len(x[1]) - len(x[0]))) / (len(x[0]) + len(x[1])),
    )  # This focuses on equal sized chunks


def merge_pair(a, b, splits, word_freqs):

    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def get_alphabet(vocab):
    alphabet = []
    for word in vocab:
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)

    alphabet.extend(["<PAD>", "<SOL>", "<EOL>", "[", "]"])
    return alphabet


def flatten(*A):
    for a in A:
        if isinstance(a, list):
            yield from flatten(*a)
        else:
            yield a
