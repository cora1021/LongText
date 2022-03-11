import argparse
parser = argparse.ArgumentParser(description='20news')

parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--model', choices=['bert', 'checkpoint','simcse'], required=True)
parser.add_argument('--seed', type=int, default=666)

args = parser.parse_args()
import torch
from torch.utils.data import (TensorDataset, DataLoader,
							  RandomSampler, SequentialSampler)

from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from distutils.version import LooseVersion as LV
from sklearn.model_selection import train_test_split
import io, sys, os, datetime
import pickle
import numpy as np
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'./logging/{args.model}')
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def set_seed(seed):
	random.seed(seed) 
	np.random.seed(seed) 
	torch.manual_seed(seed) 
	if torch.cuda.is_available(): 
		torch.cuda.manual_seed_all(seed) 
# set_seed(args.seed)
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

DEVICE = torch.device('cuda')

# ## 20 Newsgroups data set
#
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

# sample 1/10 data
# c = list(zip(train_data, train_label))
# random.shuffle(c)
# sample_data, sample_label = zip(*c)
# length = int(len(sample_data)/10)
# train_data = sample_data[:length]
# train_label = sample_label[:length]
# print(len(train_data))
# print(len(train_label))

if args.model == 'checkpoint':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	checkpoint = "/home/yujie.wang/treeloss/NLP/long_text/checkpoint"#/pytorch_model.bin"
	model = BertForSequenceClassification.from_pretrained(checkpoint,num_labels=20)
if args.model == 'bert':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	BERTMODEL='bert-base-uncased'
	model = BertForSequenceClassification.from_pretrained(BERTMODEL,num_labels=20)
if args.model == 'simcse':
	tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
	model =  BertForSequenceClassification.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", num_labels=20)
model.cuda()

(test_input, validation_input,
test_labels, validation_labels) =  train_test_split(test_data, test_label,random_state=42,
													 test_size=0.2)

WARMUP_STEPS =int(0.2*len(train_data)/args.batch_size)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in model.named_parameters()
				if not any(nd in n for nd in no_decay)],
	 'weight_decay': args.weight_decay},
	{'params': [p for n, p in model.named_parameters()
				if any(nd in n for nd in no_decay)],
	 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
											num_warmup_steps=WARMUP_STEPS,
											num_training_steps=len(train_data)/args.batch_size*args.epochs)

def train(train_data, train_label, epoch, loss_vector=None, log_interval=200):
	# Set model to training mode
	model.train()
	step = 0
	for batch in batchify(train_data, train_label, args.batch_size):
		text_batch, label_batch = batch
		train_inputs = tokenizer(text_batch, max_length=512, truncation=True,padding=True, return_tensors="pt")

		for k,v in train_inputs.items():
			train_inputs[k] = v.to(DEVICE)

		label_batch = torch.LongTensor(label_batch).to(DEVICE)
		optimizer.zero_grad()

		outputs = model(**train_inputs, labels=label_batch)

		loss = outputs[0]
		if loss_vector is not None:
			loss_vector.append(loss.item())

		# Backward pass
		loss.backward()

		# Update weights
		optimizer.step()
		scheduler.step()
		step += 1
		if step % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, step * args.batch_size,
					len(train_data),
					100. * step / len(train_data), loss))
	return loss

def evaluate(data, label):

	model.eval()

	n_correct, n_all = 0, 0
	for batch in batchify(data, label, args.batch_size):
		text_batch, label_batch = batch
		test_inputs = tokenizer(text_batch, max_length=512, truncation=True,padding=True, return_tensors="pt")
		with torch.no_grad():
			output = model(**test_inputs)
			logits = output[0]

		logits = logits.detach().cpu().numpy()
		predictions = np.argmax(logits, axis=1)

		labels = np.array(label_batch)
		n_correct += np.sum(predictions == labels)
		n_all += len(labels)

	print('Accuracy: [{}/{}] {:.4f}\n'.format(n_correct,
												n_all,
												n_correct/n_all))
	return n_correct/n_all

train_lossv = []
for epoch in tqdm(range(1, args.epochs + 1)):
	train_loss = train(train_data, train_label, epoch, train_lossv)
	writer.add_scalar('train loss: ', train_loss, epoch)
	print('\nValidation set:')
	valid_accuracy = evaluate(validation_input, validation_labels)
	writer.add_scalar('valid accuarcy: ', valid_accuracy, epoch)
	print('Test set:')
	test_accuracy = evaluate(test_input, test_labels)
	writer.add_scalar('test Accuarcy: ', test_accuracy, epoch)
