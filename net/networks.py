import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

import einops
from einops.layers.torch import Reduce
import torch
import torch.nn as nn


# Creating a CNN class
class Snoap_CNN(nn.Module):
	#  Determine what layers and their order in CNN object
	def __init__(self, num_classes):
		super(Snoap_CNN, self).__init__()
		filter_size = 23

		self.conv_layer1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=filter_size, padding=11)
		self.batch_norm1 = nn.BatchNorm1d(16)
		self.ReLU1 = nn.ReLU()
		self.max_pool1 = nn.MaxPool1d(kernel_size=filter_size, stride = 2, padding = 11)

		self.conv_layer2 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=filter_size, padding = 11)
		self.batch_norm2 = nn.BatchNorm1d(24)
		self.ReLU2 = nn.ReLU()
		self.max_pool2 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer3 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=filter_size, padding = 11)
		self.batch_norm3 = nn.BatchNorm1d(32)
		self.ReLU3 = nn.ReLU()
		self.max_pool3 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding=11)

		self.conv_layer4 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=filter_size, padding = 11)
		self.batch_norm4 = nn.BatchNorm1d(48)
		self.ReLU4 = nn.ReLU()
		self.max_pool4 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer5 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=filter_size, padding = 11)
		self.batch_norm5 = nn.BatchNorm1d(64)
		self.ReLU5 = nn.ReLU()
		self.max_pool5 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer6 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=filter_size, padding = 11)
		self.batch_norm6 = nn.BatchNorm1d(96)
		self.ReLU6 = nn.ReLU()

		self.avgpool = nn.AvgPool1d(kernel_size=32,stride=2,padding = 0)
		self.drop = nn.Dropout(0.5)
		self.fc1 = nn.Linear(96, num_classes)
		self.softmax = nn.Softmax(dim=1)

	# Progresses data across layers
	def forward(self, x):


		out = self.conv_layer1(x)
		out = self.batch_norm1(out)
		out = self.ReLU1(out)
		out = self.max_pool1(out)


		out = self.conv_layer2(out)
		out = self.batch_norm2(out)
		out = self.ReLU2(out)
		out = self.max_pool2(out)


		out = self.conv_layer3(out)
		out = self.batch_norm3(out)
		out = self.ReLU3(out)
		out = self.max_pool3(out)


		out = self.conv_layer4(out)
		out = self.batch_norm4(out)
		out = self.ReLU4(out)
		out = self.max_pool4(out)


		out = self.conv_layer5(out)
		out = self.batch_norm5(out)
		out = self.ReLU5(out)
		out = self.max_pool5(out)

		out = self.conv_layer6(out)
		out = self.batch_norm6(out)
		out = self.ReLU6(out)

		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.drop(out)
		out = self.fc1(out)
		out = self.softmax(out)

		return out

class ResidualUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResidualUnit, self).__init__()
		filter_size = 23
		self.conv1 = nn.Conv1d(32, 32, kernel_size=filter_size,
		                             padding=11)
		self.bn1 = nn.BatchNorm1d(32)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(32, 32, kernel_size=filter_size,
		                       padding =11)
		self.bn2 = nn.BatchNorm1d(32)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out += residual  # Skip connection
		return out


class ResidualStack(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResidualStack, self).__init__()
		self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
		self.bn = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU()
		self.residual_units = nn.Sequential(
			ResidualUnit(in_channels, out_channels),
			ResidualUnit(in_channels, out_channels)
		)
		self.max_pool = nn.MaxPool1d(kernel_size=2)

	def forward(self, x):
		print("x {}".format(x.size()))
		out = self.conv1x1(x)
		print("conv1x1 {}".format(out.size()))
		out = self.bn(out)
		out = self.relu(out)
		out = self.residual_units(out)
		out = self.max_pool(out)
		print("output {}".format(out.size()))
		return out


class Snoap_ResidualNN(nn.Module):
	def __init__(self, num_classes=10):
		super(Snoap_ResidualNN, self).__init__()
		self.residual_stacks = nn.Sequential(
			ResidualStack(2, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32)
		)
		self.final_layers = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(32 * 16, 128),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(128, 128),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(128, num_classes),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		out = self.residual_stacks(x)
		out = out.view(out.size(0), -1)
		print("res_stack out {}".format(out.size()))
		out = self.final_layers(out)
		return out


class EncoderLayer(nn.Module):
	def __init__(self, embedding_dim, num_heads, dropout_rate=0.1):
		super(EncoderLayer, self).__init__()
		self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate)
		self.dropout1 = nn.Dropout(dropout_rate)
		self.layer_norm1 = nn.LayerNorm(embedding_dim)
		self.feedforward = nn.Sequential(
			nn.Linear(embedding_dim, 4 * embedding_dim),
			nn.ReLU(),
			nn.Linear(4 * embedding_dim, embedding_dim)
		)
		self.dropout2 = nn.Dropout(dropout_rate)
		self.layer_norm2 = nn.LayerNorm(embedding_dim)

	def forward(self, src, src_mask=None):
		src2, _ = self.attention(src, src, src, attn_mask=src_mask)
		src = src + self.dropout1(src2)
		src = self.layer_norm1(src)
		src2 = self.feedforward(src)
		src = src + self.dropout2(src2)
		src = self.layer_norm2(src)
		return src


class PositionalEncoding(nn.Module):
	def __init__(self, embedding_dim, max_seq_length=1024):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=0.1)

		# Compute the positional encodings in advance
		pe = torch.zeros(max_seq_length, embedding_dim)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (
					-torch.log(torch.tensor(10000.0)) / embedding_dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # Add batch dimension
		self.register_buffer('pe', pe)

	def forward(self, x):
		# Adjust the positional encodings to match the input tensor size
		pe = self.pe[:, :x.size(1)]  # Trim or repeat positional encodings as needed
		print("Shape of x:", x.shape)
		print("Shape of pe:", pe.shape)
		x = x + pe
		return self.dropout(x)


class Base_Transformer(nn.Module):
	def __init__(self, num_layers, embedding_dim, num_heads, dropout_rate=0.1):
		super(Base_Transformer, self).__init__()
		self.embedding_dim = embedding_dim
		self.pos_encoder = PositionalEncoding(embedding_dim)
		self.layers = nn.ModuleList(
			[EncoderLayer(embedding_dim, num_heads, dropout_rate) for _ in range(num_layers)])
		self.norm = nn.LayerNorm(embedding_dim)

	def forward(self, src, src_mask=None):
		src = self.pos_encoder(src)
		for layer in self.layers:
			src = layer(src, src_mask)
		return self.norm(src)





class LinearEmbedding(nn.Sequential):

	def __init__(self, input_channels, output_channels) -> None:
		super().__init__(*[
			nn.Linear(input_channels, output_channels),
			nn.LayerNorm(output_channels),
			nn.GELU()
		])
		self.cls_token = nn.Parameter(torch.randn(1, output_channels))

	def forward(self, x):
		embedded = super().forward(x)
		return torch.cat([einops.repeat(self.cls_token, "n e -> b n e", b=x.shape[0]), embedded],
		                 dim=1)


class MLP(nn.Sequential):
	def __init__(self, input_channels, expansion=4):
		super().__init__(*[
			nn.Linear(input_channels, input_channels * expansion),
			nn.GELU(),
			nn.Linear(input_channels * expansion, input_channels)
		])


class ResidualAdd(torch.nn.Module):
	def __init__(self, block):
		super().__init__()
		self.block = block

	def forward(self, x):
		return x + self.block(x)


class MultiHeadAttention(torch.nn.Module):
	def __init__(self, embed_size, num_heads, attention_store=None):
		super().__init__()
		self.queries_projection = nn.Linear(embed_size, embed_size)
		self.values_projection = nn.Linear(embed_size, embed_size)
		self.keys_projection = nn.Linear(embed_size, embed_size)
		self.final_projection = nn.Linear(embed_size, embed_size)
		self.embed_size = embed_size
		self.num_heads = num_heads

	def forward(self, x):
		assert len(x.shape) == 3
		keys = self.keys_projection(x)
		values = self.values_projection(x)
		queries = self.queries_projection(x)
		keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
		queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
		values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
		energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
		divider = sqrt(self.embed_size)
		mh_out = torch.softmax(energy_term, -1)
		out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
		out = einops.rearrange(out, "b n h e -> b n (h e)")
		return self.final_projection(out)


class TransformerEncoderLayer(torch.nn.Sequential):
	def __init__(self, embed_size=768, expansion=4, num_heads=8, dropout=0.1):
		super(TransformerEncoderLayer, self).__init__(
			*[
				ResidualAdd(nn.Sequential(*[
					nn.LayerNorm(embed_size),
					MultiHeadAttention(embed_size, num_heads),
					nn.Dropout(dropout)
				])),
				ResidualAdd(nn.Sequential(*[
					nn.LayerNorm(embed_size),
					MLP(embed_size, expansion),
					nn.Dropout(dropout)
				]))
			]
		)


class Classifier(nn.Sequential):
	def __init__(self, embed_size, num_classes):
		super().__init__(*[
			Reduce("b n e -> b e", reduction="mean"),
			nn.Linear(embed_size, embed_size),
			nn.LayerNorm(embed_size),
			nn.Linear(embed_size, num_classes)
		])


class ECGformer(nn.Module):

	def __init__(self, num_layers, signal_length, num_classes, input_channels, embed_size,
	             num_heads, expansion) -> None:
		super().__init__()
		self.encoder = nn.ModuleList([TransformerEncoderLayer(
			embed_size=embed_size, num_heads=num_heads, expansion=expansion) for _ in
			range(num_layers)])
		self.classifier = Classifier(embed_size, num_classes)
		self.positional_encoding = nn.Parameter(torch.randn(signal_length + 1, embed_size))
		self.embedding = LinearEmbedding(input_channels, embed_size)

	def forward(self, x):
		embedded = self.embedding(x)

		for layer in self.encoder:
			embedded = layer(embedded + self.positional_encoding)

		return self.classifier(embedded)


class LinearEmbedding2D(nn.Sequential):
	def __init__(self, input_height, input_width, input_channels, output_channels):
		super().__init__(
			nn.Linear(input_height * input_width * input_channels, output_channels),
			nn.LayerNorm(output_channels),
			nn.GELU()
		)
		self.cls_token = nn.Parameter(torch.randn(1, output_channels))

	def forward(self, x):
		b, c, h, w = x.shape
		x = x.view(b, c * h * w)
		embedded = super().forward(x)
		return torch.cat([einops.repeat(self.cls_token, "n e -> b n e", b=b), embedded], dim=1)


class ECGformer2D(nn.Module):
	def __init__(self, num_layers, input_height, input_width, num_classes, input_channels,
	             embed_size, num_heads, expansion):
		super().__init__()
		self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_size=embed_size,
		                                                      num_heads=num_heads,
		                                                      expansion=expansion) for _ in
		                              range(num_layers)])
		self.classifier = Classifier(embed_size, num_classes)
		self.positional_encoding_height = nn.Parameter(torch.randn(input_height + 1, embed_size))
		self.positional_encoding_width = nn.Parameter(torch.randn(input_width + 1, embed_size))
		self.embedding = LinearEmbedding2D(input_height, input_width, input_channels, embed_size)

	def forward(self, x):
		embedded = self.embedding(x)
		b, n, _ = embedded.shape
		h = self.positional_encoding_height.unsqueeze(1).repeat(1, n, 1)
		w = self.positional_encoding_width.unsqueeze(0).repeat(b, n, 1)
		positional_encoding = h + w

		for layer in self.encoder:
			embedded = layer(embedded + positional_encoding)

		return self.classifier(embedded)

