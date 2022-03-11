import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import torch
import json
import numpy as np
import random
from tqdm import tqdm
from nltk import WordNetLemmatizer
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from tensorboardX import SummaryWriter
LEMMATIZER = WordNetLemmatizer()
def get_device(gpu):
    return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

def set_logger(path):
	tensorboard = SummaryWriter(path)
	return tensorboard

def update_logger(logger, losses=None, global_step=0):
	if losses is not None:
		for key, val in losses.items():
			logger.add_scalar('train/'+key, val, global_step)


def dataset_reader(path, label_dict, token_level=False):

	print(f'Read data from {path}')
	
	data = []
	with open(path, "r", encoding='utf8') as f:
		lines = f.readlines()
		for line in lines:
			line = json.loads(line)
			sentence = line['sentences'][0]
			spans = []
			ner_labels = []
			for entity in line['ner'][0]:
				spans.append((entity[0], entity[1]))
				ner_labels.append(entity[2])

			if len(ner_labels)==0:
				continue
			data.append([sentence, spans, ner_labels])
	if token_level:
		return data

	#phrase_set = set()

	data_processed = []
	for line in data:

		sentence, spans, ner_labels = line

		text = ' '.join(sentence)

		for span, label in zip(spans, ner_labels):

			if label not in label_dict:
				continue
			span_text = ' '.join(sentence[span[0]:span[1]+1])
			span_lemma_text = ' '.join([LEMMATIZER.lemmatize(word) for word in sentence[span[0]:span[1]+1]])

			# if span_text.lower() in phrase_set:
			# 	continue
			# else:
			# 	phrase_set.add(span_text.lower())

			span_start = text.find(span_text)
			span_end = span_start+len(span_text)
		
			data_processed.append({'text': text, 'span': [(span_start, span_end)], 'label': label_dict[label], 'span_lemma': span_lemma_text})

	print(f'Read {len(data_processed)} instances from dataset CoNLL2003.')
	return data_processed

def get_rankings(scores, positive_ratio = 0.8):
	'''
	scores: (samples, class_num)
	'''
	class_num = scores.shape[-1]
	rankings = (-scores).argsort(axis=0) #(samples, class_num)
	rankings = rankings[:int(len(rankings) * 1.0 / class_num * positive_ratio)]

	return rankings

def get_data(rankings, negative_numbers = 10, in_batch=False):
	'''
	rankings: (samples, class_num)
	'''
	assert rankings.shape[0]>1 and rankings.shape[1]>1

	data = []

	for i in range(rankings.shape[0]):
		for j in range(rankings.shape[1]):

			anchor = rankings[i][j]

			positive = np.random.choice(rankings[:, j])
			while positive == anchor:
				positive = np.random.choice(rankings[:, j])

			if in_batch:

				data_line = [anchor, positive]

			else:
				negative_list = []
				while len(negative_list) < negative_numbers:
					for k in range(rankings.shape[1]):

						if k!=j:
							negative = np.random.choice(rankings[:, k])
							negative_list.append(negative)

				data_line = [anchor] + [positive] + negative_list #[anchor, postive, negative, negative....]

			data.append(data_line)

	random.shuffle(data)
	print(f'Generate {len(data)} contrastive training instances.')

	return data

class Confusion(object):
	"""
	column of confusion matrix: predicted index
	row of confusion matrix: target index
	"""
	def __init__(self, k, normalized = False):
		super(Confusion, self).__init__()
		self.k = k
		self.conf = torch.LongTensor(k,k)
		self.normalized = normalized
		self.reset()

	def reset(self):
		self.conf.fill_(0)
		self.gt_n_cluster = None

	def cuda(self):
		self.conf = self.conf.cuda()

	def add(self, output, target):
		output = output.squeeze()
		target = target.squeeze()
		assert output.size(0) == target.size(0), \
				'number of targets and outputs do not match'
		if output.ndimension()>1: #it is the raw probabilities over classes
			assert output.size(1) == self.conf.size(0), \
				'number of outputs does not match size of confusion matrix'
		
			_,pred = output.max(1) #find the predicted class
		else: #it is already the predicted class
			pred = output
		indices = (target*self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
		ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
		self._conf_flat = self.conf.view(-1)
		self._conf_flat.index_add_(0, indices, ones)

	def classIoU(self,ignore_last=False):
		confusion_tensor = self.conf
		if ignore_last:
			confusion_tensor = self.conf.narrow(0,0,self.k-1).narrow(1,0,self.k-1)
		union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
		acc = confusion_tensor.diag().float().view(-1).div(union.float()+1)
		return acc
		
	def recall(self,clsId):
		i = clsId
		TP = self.conf[i,i].sum().item()
		TPuFN = self.conf[i,:].sum().item()
		if TPuFN==0:
			return 0
		return float(TP)/TPuFN
		
	def precision(self,clsId):
		i = clsId
		TP = self.conf[i,i].sum().item()
		TPuFP = self.conf[:,i].sum().item()
		if TPuFP==0:
			return 0
		return float(TP)/TPuFP
		
	def f1score(self,clsId):
		r = self.recall(clsId)
		p = self.precision(clsId)
		print("classID:{}, precision:{:.4f}, recall:{:.4f}".format(clsId, p, r))
		if (p+r)==0:
			return 0
		return 2*float(p*r)/(p+r)
		
	def acc(self):
		TP = self.conf.diag().sum().item()
		total = self.conf.sum().item()
		if total==0:
			return 0
		return float(TP)/total
		
	def optimal_assignment(self,gt_n_cluster=None,assign=None):
		if assign is None:
			mat = -self.conf.cpu().numpy() #hungaian finds the minimum cost
			r,assign = hungarian(mat)
		self.conf = self.conf[:,assign]
		self.gt_n_cluster = gt_n_cluster
		return assign
		
	def show(self,width=6,row_labels=None,column_labels=None):
		print("Confusion Matrix:")
		conf = self.conf
		rows = self.gt_n_cluster or conf.size(0)
		cols = conf.size(1)
		if column_labels is not None:
			print(("%" + str(width) + "s") % '', end='')
			for c in column_labels:
				print(("%" + str(width) + "s") % c, end='')
			print('')
		for i in range(0,rows):
			if row_labels is not None:
				print(("%" + str(width) + "s|") % row_labels[i], end='')
			for j in range(0,cols):
				print(("%"+str(width)+".d")%conf[i,j],end='')
			print('')
		
	def conf2label(self):
		conf=self.conf
		gt_classes_count=conf.sum(1).squeeze()
		n_sample = gt_classes_count.sum().item()
		gt_label = torch.zeros(n_sample)
		pred_label = torch.zeros(n_sample)
		cur_idx = 0
		for c in range(conf.size(0)):
			if gt_classes_count[c]>0:
				gt_label[cur_idx:cur_idx+gt_classes_count[c]].fill_(c)
			for p in range(conf.size(1)):
				if conf[c][p]>0:
					pred_label[cur_idx:cur_idx+conf[c][p]].fill_(p)
				cur_idx = cur_idx + conf[c][p]
		return gt_label,pred_label
	
	def clusterscores(self):
		target,pred = self.conf2label()
		NMI = normalized_mutual_info_score(target,pred)
		ARI = adjusted_rand_score(target,pred)
		AMI = adjusted_mutual_info_score(target,pred)
		return {'NMI':NMI,'ARI':ARI,'AMI':AMI}

def get_kmeans(all_features, all_labels, num_classes):

    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)
    print('Clustering with kmeans...')
    # Perform kmean clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_features)
    cluster_assignment = clustering_model.labels_

    score_factor = np.matmul(all_features, clustering_model.cluster_centers_.transpose())
    score_cosine = cosine(all_features, clustering_model.cluster_centers_)

    if all_labels is None:
        return score_factor, score_cosine, clustering_model.cluster_centers_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, centers:{}, true_labels:{}, pred_labels:{}".format(all_features.shape, clustering_model.cluster_centers_.shape, len(true_labels), len(pred_labels)))
    
    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    
    confusion_factor = Confusion(num_classes)
    score_factor = np.matmul(all_features, clustering_model.cluster_centers_.transpose())
    pred_labels_factor = score_factor.argmax(axis=-1)
    pred_labels_factor = torch.tensor(pred_labels_factor)
    confusion_factor.add(pred_labels_factor, true_labels)
    confusion_factor.optimal_assignment(num_classes)

    confusion_cosine = Confusion(num_classes)
    score_cosine = cosine(all_features, clustering_model.cluster_centers_)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    pred_labels_cosine = torch.tensor(pred_labels_cosine)
    confusion_cosine.add(pred_labels_cosine, true_labels)
    confusion_cosine.optimal_assignment(num_classes)

    print("Clustering iterations:{}, L2 ACC:{:.3f}, Inner ACC:{:.3f}, Cosine ACC:{:.3f}".format(clustering_model.n_iter_, confusion.acc(), confusion_factor.acc(), confusion_cosine.acc()))
    print('L2 Clustering scores:',confusion.clusterscores())
    print('Inner Clustering scores:',confusion_factor.clusterscores()) 
    print('Cosine Clustering scores:',confusion_cosine.clusterscores()) 
    return score_factor, score_cosine, clustering_model.cluster_centers_

def get_kmeans_centers(all_features, all_labels, num_classes):

    _, _, centers = get_kmeans(all_features, all_labels, num_classes)

    return centers

def get_kmeans_prediction_and_centers(all_features, all_labels, num_classes):

    _, score_cosine, centers = get_kmeans(all_features, all_labels, num_classes)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    return pred_labels_cosine, centers

def get_metric(features, centers, labels, num_classes):

    normalized_features = preprocessing.normalize(np.concatenate((centers, features), axis=0))
    centers, features = normalized_features[:num_classes], normalized_features[num_classes:]

    confusion_factor = Confusion(num_classes)
    score_factor = np.matmul(features, centers.transpose())
    pred_labels_factor = score_factor.argmax(axis=-1)
    pred_labels_factor = torch.tensor(pred_labels_factor)
    confusion_factor.add(pred_labels_factor, labels)
    confusion_factor.optimal_assignment(num_classes)

    confusion_cosine = Confusion(num_classes)
    score_cosine = cosine(features, centers)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    pred_labels_cosine = torch.tensor(pred_labels_cosine)
    confusion_cosine.add(pred_labels_cosine, labels)
    confusion_cosine.optimal_assignment(num_classes)

    print("Inner ACC:{:.3f}, Cosine ACC:{:.3f}".format(confusion_factor.acc(), confusion_cosine.acc()))
    print('Inner Clustering scores:', confusion_factor.clusterscores()) 
    print('Cosine Clustering scores:',confusion_cosine.clusterscores())


def get_kmeans_score(all_features, num_classes):

    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)

    clustering_model = KMeans(n_clusters=num_classes)
    labels = clustering_model.fit_predict(all_features)
    silhouette = silhouette_score(all_features, labels)

    return silhouette