import os
import argparse
import time
import math, random
import numpy as np
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import wandb
import sentencepiece as spm
from env import *
import jericho
from jericho.template_action_generator import TemplateActionGenerator

class TDQN_Trainer(object):
    def __init__(self, args):
        self.args = args

        self.log_freq = args.log_freq
        self.update_freq = args.update_freq_td
        self.update_freq_tar = args.update_freq_tar
        self.filename = 'random'+args.rom_path + str(args.run_number)
        wandb.init(project="my-project", name=self.filename)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.binding = jericho.load_bindings(args.rom_path)
        self.vocab_act, self.vocab_act_rev = self.load_vocab_act(args.rom_path)
        vocab_size = len(self.sp)
        self.vocab_size_act = len(self.vocab_act.keys())

        self.template_generator = TemplateActionGenerator(self.binding)
        self.template_size = len(self.template_generator.templates)

        self.num_steps = args.steps
    

    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.vocab_act[o1_id])
        else:
            return template_str.replace('OBJ', self.vocab_act[o1_id], 1)\
                               .replace('OBJ', self.vocab_act[o2_id], 1)


    def load_vocab_act(self, rom_path):
        #loading vocab directly from Jericho
        env = FrotzEnv(rom_path)
        vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
        vocab[0] = ' '
        vocab[1] = '<s>'
        vocab_rev = {v: idx for idx, v in vocab.items()}
        env.close()
        return vocab, vocab_rev
    


    

    def state_rep_generator(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            state_description = state_description.replace(rm, '')

        state_description = state_description.split('|')

        ret = [self.sp.encode_as_ids('<s>' + s_desc + '</s>') for s_desc in state_description]

        return pad_sequences(ret, maxlen=self.args.max_seq_len)


    def train(self):
            start = time.time()
            env = JerichoEnv(self.args.rom_path, 0, self.vocab_act_rev,
                             self.args.env_step_limit)
            env.create()
            episode = 1
            episode_score = np.array([])
            episode_index = 0
            state_text, info = env.reset()
            state_rep = self.state_rep_generator(state_text)
            for frame_idx in range(1, self.num_steps + 1):
                found_valid_action = False
                while not found_valid_action:
                    template = np.random.randint(self.template_size)
                    o1 = np.random.randint(self.vocab_size_act)
                    o2 = np.random.randint(self.vocab_size_act)
                    #print(template, o1, o2)
                    action = [template, o1, o2]
                    action_str = self.tmpl_to_str(template, o1, o2)
                    next_state_text, reward, done, info = env.step(action_str)
                    if info['action_valid'] == True:
                        found_valid_action = True
                        break
                next_state_rep = self.state_rep_generator(next_state_text)
                state_text = next_state_text
                state_rep = next_state_rep
    
                if done:
                    score = info['score']
                    if episode < 11:
                        episode_score = np.append(episode_score, [score])
                    else:
                        episode_score[episode_index] = score 
                    wandb.log({'Epoch (step1)': frame_idx, 'Individual random Score': score})
                    wandb.log({'Epoch (step2)': episode, 'Average random Score': np.mean(episode_score)})
                    state_text, info = env.reset()
                    state_rep = self.state_rep_generator(state_text)
                    episode += 1
                    episode_index = (episode_index + 1)%10
            env.close()
            parameters = {
            }
            wandb.save("mymodel.h5")  
    
def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.0):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                            (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom_path', default='zork2.z5')
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='spm_models/unigram_8k.model')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--gamma', default=.95, type=float)
    parser.add_argument('--rho', default=.5, type=float)
    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--steps', default=1000000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--update_freq_td', default=4, type=int)
    parser.add_argument('--update_freq_tar', default=1000, type=int)
    parser.add_argument('--replay_buffer_size', default=100000, type=int)
    parser.add_argument('--replay_buffer_type', default='priority')
    parser.add_argument('--clip', default=40, type=float)
    parser.add_argument('--max_seq_len', default=300, type=int)
    parser.add_argument('--run_number', default = 1, type = int)
    return parser.parse_args()
    
if __name__ == "__main__":
    assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    trainer = TDQN_Trainer(args)
    trainer.train()