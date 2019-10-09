#!/usr/bin/env python
# coding: utf-8

import copy
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_ebed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_ebed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return Variable(self.encoder(self.src_embed(src), src_mask))

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return Variable(self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask))


class Generator(nn.Module):
    # 'define standard linear + softmax generation step'
    # 'map the matrix to a vector that represents the probability, so the output dimension is the size of vocab'

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    # 'produce N identical layers'
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    # core encoder is a stack of N layers
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # pass the input and mask through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    # 'a residual connection followed by a layer norm.'
    # 'note for code simplicity the norm is first as opposed to last'
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 'apply residual connection to any sublayer with the same size'
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 'follow figure 1 for connection'
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    # 'generic N layer decoder with masking'
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 'follow figure 1 for connection'
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    # 'mask out subsequent positions.'
    attn_shape = (1, size, size)
    # np.triu 返回上三角矩阵 就是把之后位置的数据mask掉
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    # 'compute scaled dot product attention'
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # masked_fill_(mask, value): 在mask值为1的位置处用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同。
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 'implement figure 2'
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 'do all the linear projections in batch from d_model-> h * d_k'
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))
                             ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # 'implement the PE function'
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 'compute the positional encodings once in log space'
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                           nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                           Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask


def run_epoch(data_iter, model, loss_compute,epoch=0):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss.detach().cpu().numpy()
        total_tokens += batch.ntokens.cpu().numpy()
        tokens += batch.ntokens.cpu().numpy()
        if i % 50 == 1:
            elapsed = time.time() - start
            print('epoch step: %d loss: %f tokens per sec: %f' % (i, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    # 'keep augmenting batch and calculate total number of tokens + padding.'
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # 'update parameters and rate'
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 'implement `lrate` above'
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    # 'implement label smoothing'
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=True))


crit = LabelSmoothing(5,0,0.1)


def loss(x):
    d=x+3*1
    predict=torch.FloatTensor([[0,x/d,1/d,1/d,1/d],])
    return crit(Variable(predict.log()),Variable(torch.LongTensor([1]))).item()


def data_gen(V,batch,nbatches):
    # 'generate random data for a src_tgt copy task'
    for i in  range(nbatches):
        data=torch.from_numpy(np.random.randint(1,V,size=(batch,10)))
        data[:,0]=1
        src=Variable(data,requires_grad=False).long()
        tgt=Variable(data,requires_grad=False).long()
        yield Batch(src,tgt,0)


class SimpleLossCompute:
    # 'a simple loss compute and train function'
    def __init__(self,generator,criterion,opt=None):
        self.generator=generator
        self.criterion=criterion
        self.opt=opt
    
    def __call__(self,x,y,norm):
        x=self.generator(x)
        loss=self.criterion(x.contiguous().view(-1,x.size(-1)),y.contiguous().view(-1))/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()*norm.float()


# this code predicts a translation using greedy decoder for simplicity.
def greedy_decode(model,src,src_mask,max_len,start_symbol):
    memory=model.encode(src,src_mask)
    ys=torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out=model.decode(memory,src_mask,Variable(ys),Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob=model.generator(out[:,-1])
        _,next_word=torch.max(prob,dim=1)
        next_word=next_word.data[0]
        ys=torch.cat([ys,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=1)
    return ys


# for data loading
from torchtext import data,datasets

if True:
    import spacy
    spacy_de=spacy.load('de')
    spacy_en=spacy.load('en')
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    BOS_WORD='<s>'
    EOS_WORD='</s>'
    BLANK_WORD='<blank>'
    SRC=data.Field(tokenize=tokenize_de,pad_token=BLANK_WORD)
    TGT=data.Field(tokenize=tokenize_en,init_token=BOS_WORD,eos_token=EOS_WORD,pad_token=BLANK_WORD)
    
    MAX_LEN=100
    train,val,test=datasets.IWSLT.splits(
    exts=('.de','.en'),fields=(SRC,TGT),filter_pred=lambda x:len(vars(x)['trg'])<=MAX_LEN
    )
    
    MIN_FREQ=2
    SRC.build_vocab(train.src,min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg,min_freq=MIN_FREQ)


# Batching matters a ton for speed. We want to have very evenly divided batches,
# with absolutely minimal padding. To do this we have to hack a bit around the default
# torchtext batching. This code patches their default batching to make sure
# we search over enough sentences to find tight batches.

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d,random_shuffler):
                for p in data.batch(d,self.batch_size*100):
                    p_batch=data.batch(
                        sorted(p,key=self.sort_key),
                        self.batch_size,self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches=pool(self.data(),self.random_shuffler)
            
        else:
            self.batches=[]
            for b in data.batch(self.data(),self.batch_size,self.batch_size_fn):
                self.batches.append(sorted(b,key=self.sort_key))


def rebatch(pad_idx,batch):
    # 'fix order in torchtext to match ours'
    src,trg=batch.src.transpose(0,1),batch.trg.transpose(0,1)
    return Batch(src,trg,pad_idx)


# MultiGPU
class MultiGPULossCompute:
    # 'a multi-gpu loss compute and train function'
    def __init__(self,generator,criterion,devices,opt=None,chunk_size=5):
        # send out to different gpus.
        self.generator=generator
        self.criterion=nn.parallel.replicate(criterion,devices=devices)
        self.opt=opt
        self.devices=devices
        self.chunk_size=chunk_size
        
    def __call__(self,out,targets,normalize):
        total=0.0
        generator=nn.parallel.replicate(self.generator,devices=self.devices)
        out_scatter=nn.parallel.scatter(out,target_gpus=self.devices)
        out_grad=[[] for _ in out_scatter]
        targets=nn.parallel.scatter(targets,target_gpus=self.devices)
        
        # divide generating into chunks. 
        chunk_size=self.chunk_size
        for i in range(0,out_scatter[0].size(1),chunk_size):
            out_column=[[Variable(o[:,i:i+chunk_size].data,requires_grad=(self.opt is not None))] for o in out_scatter]
            gen=nn.parallel.parallel_apply(generator,out_column)
        
            #compute loss
            y=[(g.contiguous().view(-1,g.size(-1)),t[:,i:i+chunk_size].contiguous().view(-1)) for g,t in zip(gen,targets)]
            loss=nn.parallel.parallel_apply(self.criterion,y)

            #sum and normalize loss
            l=nn.parallel.gather(loss,target_device=self.devices[0])
            l=l.sum()[0]/normalize
            total+=l.item()

            #backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j,l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        #backprop all loss through transformer
        if self.opt is not None:
            out_grad=[Variable(torch.cat(og,dim=1),requires_grad=True) for og in out_grad]
            o1=Variable(out,requires_grad=True)
            o2=nn.parallel.gather(out_grad,target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total*normalize


# gpus to use.
devices=[0,1,2,3]
if torch.cuda.is_available():
    pad_idx=TGT.vocab.stoi['<blank>']
    model=make_model(len(SRC.vocab),len(TGT.vocab),N=6)
    model.cuda()
    criterion=LabelSmoothing(size=len(TGT.vocab),padding_idx=pad_idx,smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE=20
    train_iter=MyIterator(train,batch_size=BATCH_SIZE,device=torch.device('cuda'),repeat=False,sort_key=lambda x:(len(x.src),len(x.trg)),batch_size_fn=batch_size_fn,train=True)
    valid_iter=MyIterator(val,batch_size=BATCH_SIZE,device=torch.device('cuda'),repeat=False,sort_key=lambda x:(len(x.src),len(x.trg)),batch_size_fn=batch_size_fn,train=False)
    model_par=nn.DataParallel(model,device_ids=devices)

    model_opt=NoamOpt(model.src_embed[0].d_model,1,2000,torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx,b) for b in train_iter),model_par,MultiGPULossCompute(model.generator,criterion,devices=devices,opt=model_opt))
        model_par.eval()
        run_epoch((rebatch(pad_idx,b) for b in valid_iter),model_par,MultiGPULossCompute(model.generator,criterion,devices=devices,opt=None))
        print(loss)

    for i,batch in enumerate(valid_iter):
        src=batch.src.transpose(0,1)[:1].cuda()
        src_mask=(src!=SRC.vocab.stoi['<blank>']).unsqueeze(-2)
        out=greedy_decode(model,src,src_mask,max_len=60,start_symbol=TGT.vocab.stoi['<s>'])
        print('translation:',end='\t')
        for i in range(1,out.size(1)):
            sym=TGT.vocab.itos[out[0,i]]
            if sym=='</s>':
                break
            print(sum,end=' ')

        print()
        print('target:',end='\t')
        for i in range(1,batch.trg.size(0)):
            sym=TGT.vocab.itos[batch.trg.data[i,0]]
            if sym=='</s>':
                break

            print(sym,end=' ')
        print()
        break


