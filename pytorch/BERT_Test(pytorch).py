import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# 定义原始数据集
text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)

#  input Embedding
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]':0, '[CLS]':1, '[SEP]':2, '[MASK]':3}

for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()

for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

# Data mask
def make_data(sentence):
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        token_a_index, token_b_index = randrange(len(sentence)), randrange(len(sentence))
        token_a, token_b = token_list[sentence[token_a_index]], token_list[sentence[token_b_index]]
        input_ids = [word2idx['[CLS]']] + token_a + [word2idx['[SEP]']] + token_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)

        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 确定了本次输入数据中实际要掩码的词的大致数量
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # 寻找可以mask的位置
        shuffle(cand_maked_pos)  #随机打乱顺序，使得掩码位置的选择更加随机，避免有规律的掩码情况，增加数据的随机性和模型训练的难度与泛化能力。
        masked_tokens, masked_pos = [], []  #存储被掩码的词的原始索引编号以及被掩码的位置信息。
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace
        
        # 数据填充操作
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 额外的数据填充操作（针对被掩码的词标识符和位置信息）
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_pos.extend([0] * n_pad)
            masked_tokens.extend([0] * n_pad)
        
        # 根据句子关系添加数据到批次并更新计数器
        if token_a_index + 1 == token_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif token_a_index + 1 != token_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
        
    return batch

# 超参数设置
maxlen = 30
batch_size = 6
max_pred = 5 # max tokens of prediction
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2

# 模型构建
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    pad_atte_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_atte_mask.expand(batch_size, seq_len, seq_len)

# 激活函数
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 构建嵌入层
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  
        self.pos_embed = nn.Embedding(maxlen, d_model)  
        self.seg_embed = nn.Embedding(n_segments, d_model)  
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
                      
# 构建Transformer层
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]

# FeedForward层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

# BERT模型
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

# 创建数据加载器
batch = make_data(token_list)
loader = Data.DataLoader(batch, batch_size, shuffle=True)

# 开始训练
for epoch in range(180):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

# 测试
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)




































































