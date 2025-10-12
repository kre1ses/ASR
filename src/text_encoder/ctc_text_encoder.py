import re
from string import ascii_lowercase
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from pyctcdecode import build_ctcdecoder

import gzip
import shutil
from speechbrain.utils.data_utils import download_file # hope ts works

import torch
import numpy as np
import os
import kenlm
from multiprocessing import Pool, set_start_method

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, 
                alphabet=None, 
                bpe_use=False,
                lm_use=False,
                vocab_size=None,
                beam_size=None,
                beam_use=False,
                **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.beam_size = beam_size
        self.beam_use = beam_use
        self.lm_use = lm_use
        self.bpe_use = bpe_use

        self.path_to_bpe_text = "https://openslr.trmal.net/resources/11/librispeech-lm-norm.txt.gz"
        self.model_path = "https://openslr.trmal.net/resources/11/4-gram.arpa.gz"

        if vocab_size is not None:
            self.vocab_size = vocab_size

        if bpe_use:
            self.tokens_path = Path(os.getcwd()).absolute().resolve() / 'bpe_data/tokens.json'

            if not self.tokens_path.exists():
                self.get_tokenizer()

            self.tokenizer = Tokenizer.from_file(str(self.tokens_path))

            self.char2ind = self.tokenizer.get_vocab()
            self.ind2char = {v: k.lower() for k, v in self.char2ind.items()}
            self.char2ind = {v: k for k, v in self.ind2char.items()}
            self.vocab = [self.ind2char[ind] for ind in range(len(self.ind2char))]

        else:
            if alphabet is None:
                alphabet = list(ascii_lowercase + " ")

            self.alphabet = alphabet
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}
        
        if lm_use:
            self.get_lm()
    
    def get_lm(self):
        lm_vocab = self.vocab
        lm_vocab[0] = ""
        lm_vocab = [token.upper() for token in lm_vocab]

        path2lm = Path(os.getcwd()).absolute().resolve() / 'lm_model'
        lm_path = path2lm / '4-gram.arpa'
        gz_lm_path = path2lm / '4-gram.arpa.gz'

        if not lm_path.exists():
            if not gz_lm_path.exists():
                download_file(self.model_path, dest=gz_lm_path)

            with gzip.open(str(gz_lm_path), 'rb') as f_in:
                with open(str(lm_path), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        self.lm_decoder = build_ctcdecoder(
                lm_vocab,
                kenlm_model_path=str(lm_path)
            )
    
    def get_tokenizer(self):
        ''' Create a tokenizer with BPE model and save it to the file 'tokens.json' '''
        path2text = Path(os.getcwd()).absolute().resolve()  / 'bpe_data'

        text_path = path2text / 'librispeech-lm-norm.txt'
        gz_text_path = path2text / 'librispeech-lm-norm.txt.gz'

        if not text_path.exists():
            if not gz_text_path.exists():
                download_file(self.path_to_bpe_text, dest=gz_text_path)

            with gzip.open(str(gz_text_path), 'rb') as f_in:
                with open(str(text_path), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "'", "^", " "], vocab_size=self.vocab_size)

        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([str(text_path)], trainer)

        tokenizer.save(str(self.tokens_path))

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str: #self.ind2char
        # like in seminar but in characters
        prev_char = self.EMPTY_TOK
        result = []
        for ind in inds:
            if self.ind2char[ind] == prev_char:
                continue
            char = self.ind2char[ind]
            if char != self.EMPTY_TOK:
                result.append(char)
            prev_char = char
        return ("".join(result)).replace("'", "").strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
    
    def expand_and_merge_beams(self, 
                               dp: dict[tuple[str, str], float],
                               cur_step_prob: torch.Tensor,
                               ) -> dict[tuple[str, str], float]:
        # based on seminar
        
        new_dp = defaultdict(float)
        for (pref, prev_char), pref_proba in dp.items():
            for idx, char in enumerate(self.vocab):
                cur_proba = pref_proba * cur_step_prob[idx] # log probs
                cur_char = char

                if char == self.EMPTY_TOK:
                    cur_pref = pref
                else:
                    if prev_char != char:
                        cur_pref = pref + char
                    else:
                        cur_pref = pref

                new_dp[(cur_pref, cur_char)] += cur_proba
        return new_dp
    
    def truncate_beams(self, dp: dict[tuple[str, str], float]):
        # based on seminar
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:self.beam_size])
    
    def ctc_beam_search(self, log_probs):
        # based on seminar
        # first arg without empty token
        probs = np.exp(log_probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0
        }
        for cur_step_prob in probs:
            dp = self.expand_and_merge_beams(dp, cur_step_prob)
            dp = self.truncate_beams(dp)
        result = [(pref, prob) for (pref, _), prob in dp.items()]
        return result
    
    def ctc_lm_beam_search(self, log_probs, lengths):
        
        # log_probs = torch.nn.functional.log_softmax(probs, -1)

        logits_list = [log_probs[i][:lengths[i]].numpy() for i in range(lengths.shape[0])]

        set_start_method("fork", force=True)
        with Pool(processes=4) as pool:
            text_list = self.lm_decoder.decode_batch(logits_list=logits_list, beam_width=self.beam_size, pool = pool)

        text_list = [elem.lower().replace("'", "").replace("??", "").strip() for elem in text_list]

        return text_list
