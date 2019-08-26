# import modules
import torch
import numpy as np
import torch.nn as nn
from os.path import join
import torch.nn.functional as F

# relative imports
from base import BaseModel


def create_emb_layer(embed_file, trainable=True):
    # load weights matrix
    with open(join('embeddings', embed_file), 'r') as f:
        weights_matrix = np.loadtxt(f, delimiter=',')
    weights_matrix = torch.from_numpy(weights_matrix)

    # obtain dimensions
    n_embeds, embed_dim = weights_matrix.size()

    # initialize embedding layer
    emb_layer = nn.Embedding(n_embeds, embed_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, n_embeds, embed_dim


class AmzCNNModel(BaseModel):
    def __init__(self, num_classes=32, embed_file='amazon.glove.300.csv',
                 in_channels=1, out_channels=128, kernel_heights=[2, 3, 4],
                 keep_prob=0.6, trainable=True):
        super().__init__()

        self.embed_layer, self.n_embeds, self.embed_dim = create_emb_layer(embed_file, trainable)

        self.conv_title_1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], self.embed_dim))
        self.conv_title_2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], self.embed_dim))
        self.conv_title_3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], self.embed_dim))

        self.conv_desc_1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], self.embed_dim))
        self.conv_desc_2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], self.embed_dim))
        self.conv_desc_3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], self.embed_dim))

        self.dropout = nn.Dropout(keep_prob)
        self.label = nn.Linear(len(kernel_heights) * out_channels * 2, num_classes)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out

    def forward(self, title, desc):
        # apply embedding layer
        title = self.embed_layer(title)
        title = title.unsqueeze(1)

        desc = self.embed_layer(desc)
        desc = desc.unsqueeze(1)

        # apply conv layers
        maxout_title_1 = self.conv_block(title, self.conv_title_1)
        maxout_title_2 = self.conv_block(title, self.conv_title_2)
        maxout_title_3 = self.conv_block(title, self.conv_title_3)

        maxout_desc_1 = self.conv_block(desc, self.conv_desc_1)
        maxout_desc_2 = self.conv_block(desc, self.conv_desc_2)
        maxout_desc_3 = self.conv_block(desc, self.conv_desc_3)

        # apply dropout layer
        allout_title = torch.cat((maxout_title_1, maxout_title_2, maxout_title_3), 1)
        allout_desc = torch.cat((maxout_desc_1, maxout_desc_2, maxout_desc_3), 1)
        all_out = torch.cat((allout_title, allout_desc), 1)
        drop_out = self.dropout(all_out)
        logits = self.label(drop_out)

        return logits
