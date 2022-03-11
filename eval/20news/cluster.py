import argparse
parser = argparse.ArgumentParser(description='20news_cluster')

parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--num_classes',type=int,default=8)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--model', choices=['bert', 'checkpoint','simcse'], required=True)
parser.add_argument('--seed', type=int, default=200)

args = parser.parse_args()
import torch
from torch.utils.data import (TensorDataset, DataLoader,
							  RandomSampler, SequentialSampler)

from transformers import BertTokenizer, BertConfig
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from distutils.version import LooseVersion as LV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch
from kmeans import get_kmeans
from transformers import AutoModel, AutoTokenizer

def set_seed(seed):
	random.seed(seed) 
	np.random.seed(seed) 
	torch.manual_seed(seed) 
	if torch.cuda.is_available(): 
		torch.cuda.manual_seed_all(seed) 
set_seed(args.seed)
DEVICE = torch.device('cuda')
# ## 20 Newsgroups data set
# Next we'll load the [20 Newsgroups]
# (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
# data set.
#
# The dataset contains 20000 messages collected from 20 different
# Usenet newsgroups (1000 messages from each group):
#
# | alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
# | talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
# | talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
# | talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
# | talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale

f_train_text = open("/data/yujie.wang/nlp_data/20news/20news_train_text.pkl",'rb')
f_train_label = open("/data/yujie.wang/nlp_data/20news/20news_train_label.pkl",'rb')
f_test_text = open("/data/yujie.wang/nlp_data/20news/20news_test_text.pkl", 'rb')
f_test_label = open("/data/yujie.wang/nlp_data/20news/20news_test_label.pkl", 'rb')
train_data = pickle.load(f_train_text)
train_label = pickle.load(f_train_label)
test_data = pickle.load(f_test_text)
test_label = pickle.load(f_test_label)

print('Length of training texts:', len(train_data))
print('Length of training labels:', len(train_label))
print('Length of test texts:', len(test_data))
print('Length of test labels:', len(test_label))

text_data = train_data + test_data
label = train_label + test_label

# sample labels
# cluster = random.sample(range(0, 20), args.num_classes)
# cluster = [3, 10, 16]
cluster = [1,4,7,11,12,15,16,18]
cluster_data = []
cluster_label = []
for ele in cluster:
	indices = [i for i, x in enumerate(label) if x == ele]
	temp = [text_data[i] for i in indices]
	temp_ = [cluster.index(ele)] * len(temp)
	cluster_data.extend(temp)
	cluster_label.extend(temp_)
c = list(zip(cluster_data, cluster_label))
random.shuffle(c)
cluster_data, cluster_label = zip(*c)

def batchify(data, label, batch_size):

	batches = []
	pointer = 0
	total_num = len(data)
	while pointer < total_num:
		text_batch = []
		label_batch = []
		for data_line in data[pointer:pointer+batch_size]:
			text = data_line
			text_batch.append(text)
		
		for ele in label[pointer:pointer+batch_size]:
			temp = ele
			label_batch.append(temp)

		batches.append((text_batch, label_batch))
		pointer += batch_size

	return batches

print('Initializing BertTokenizer')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
if args.model == 'checkpoint':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	checkpoint = "/home/yujie.wang/treeloss/NLP/long_text/checkpoint"#/pytorch_model.bin"
	model = BertModel.from_pretrained(checkpoint)
if args.model == 'bert':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	BERTMODEL='bert-base-uncased'
	model = BertModel.from_pretrained(BERTMODEL)
if args.model == 'simcse':
	tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
	model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

model.cuda()
model.eval()
all_features = []
all_labels = []
with torch.no_grad():

	for batch in tqdm(batchify(cluster_data, cluster_label, args.batch_size), ncols=100, desc='Generate all features...'):

		text_batch, label_batch = batch
		
		inputs = tokenizer(text_batch, max_length=512, truncation=True,padding=True, return_tensors="pt")

		for k,v in inputs.items():
			inputs[k] = v.to(DEVICE)
		output = model(**inputs)
	
		all_features.append(output['pooler_output'].squeeze().detach().cpu())

		all_labels += label_batch

all_features = torch.cat(all_features, dim=0)
all_labels = torch.LongTensor(all_labels)

print(all_features.size())
print(all_labels.size())

score_factor, score_cosine, cluster_centers = get_kmeans(all_features, all_labels, args.num_classes)
