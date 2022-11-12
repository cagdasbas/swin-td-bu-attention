import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

	def __init__(self, in_features=1024, embedding_size=128, attention_size=1):
		super(Attention, self).__init__()
		self.L = in_features
		self.D = embedding_size
		self.K = attention_size

		self.attention = nn.Sequential(
			nn.Linear(self.L, self.D),
			nn.Tanh(),
			nn.Linear(self.D, self.K)
		)

		self.classifier = nn.Sequential(
			nn.Linear(self.L, 1),
			nn.Sigmoid()
		)

	def forward(self, features, batch_size):
		box_per_image = int(features.shape[0] / batch_size)

		reshaped = features.view([batch_size, box_per_image, features.shape[1]])

		Y_prob = torch.FloatTensor(batch_size, features.shape[1]).cuda()
		attention_scores = torch.FloatTensor(batch_size, box_per_image).cuda()

		for index in range(reshaped.shape[0]):
			reshaped_features = reshaped[index, :]
			A = self.attention(reshaped_features)  # NxK
			A = torch.transpose(A, 1, 0)  # KxN
			A = F.softmax(A, dim=1)  # softmax over N

			attention_scores[index, :] = A
			M = torch.mm(A, reshaped_features)  # KxL

			Y_prob[index, :] = M

		return Y_prob, attention_scores

	# AUXILIARY METHODS
	def calculate_classification_error(self, X, Y):
		Y = Y.float()
		_, Y_hat, _ = self.forward(X)
		error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

		return error, Y_hat

	def calculate_objective(self, X, Y):
		Y = Y.float()
		Y_prob, _, A = self.forward(X)
		Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
		neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

		return neg_log_likelihood, A
