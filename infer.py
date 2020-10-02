# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import yaml
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from hydra.utils import instantiate

from utils import show_tensor_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    return parser.parse_args()


def collect(dataloader, encoder, device):
    ''' Encodes features by class label '''
    features = {}
    for (x, _, inst, _) in tqdm(dataloader):
        x = x.to(device)
        inst = inst.to(device)
        area = inst.size(2) * inst.size(3)

        # get pooled feature map
        with torch.no_grad():
            feature_map = encoder(x, inst)

        for i in torch.unique(inst):
            label = i if i < 1000 else i // 1000
            label = int(label.flatten(0).item())

            # all indices should have same feature per class from pooling
            idx = torch.nonzero(inst == i, as_tuple=False)
            n_inst = idx.size(0)
            idx = idx[0, :]

            # retrieve corresponding encoded feature
            feature = feature_map[idx[0], :, idx[2], idx[3]].unsqueeze(0)

            # compute rate of feature appearance (in official code, they compute per block)
            block_size = 32
            rate_per_block = 32 * n_inst / area
            rate = torch.ones((1, 1), device=device).to(feature.dtype) * rate_per_block

            feature = torch.cat((feature, rate), dim=1)
            if label in features.keys():
                features[label] = torch.cat((features[label], feature), dim=0)
            else:
                features[label] = feature

    return features


def cluster(features, n_classes):
    ''' Clusters features by class label '''
    k = 10
    centroids = {}
    for label in range(n_classes):
        if label not in features.keys():
            continue
        feature = features[label]

        # thresholding by 0.5 isn't mentioned in the paper, but is present in the
        # official code repository, probably so that only frequent features are clustered
        feature = feature[feature[:, -1] > 0.5, :-1].cpu().numpy()

        if feature.shape[0]:
            n_clusters = min(feature.shape[0], k)
            kmeans = KMeans(n_clusters=n_clusters).fit(feature)
            centroids[label] = kmeans.cluster_centers_

    return centroids


def encode(label_map, centroids, n_features, device):
    # sample feature vector centroids
    b, _, h, w = label_map.shape
    feature_map = torch.zeros((b, n_features, h, w), device=device).to(label_map.dtype)

    for i in torch.unique(label_map):
        label = int(label.flatten(0).item())

        if label in centroids.keys():
            centroid_idx = random.randint(0, centroids[label].shape[0] - 1)
            idx = torch.nonzero(label_map == int(i), as_tuple=False)

            feature = torch.from_numpy(centroids[label][centroid_idx, :]).to(device)
            feature_map[idx[:, 0], :, idx[:, 2], idx[:, 3]] = feature

    return feature_map


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = instantiate(config.encoder).to(device)
    generator = instantiate(config.generator).to(device)

    state_dict = torch.load(config.resume_checkpoint)
    encoder.load_state_dict(state_dict['e_state_dict'])
    generator.load_state_dict(state_dict['g_state_dict'])

    train_dataloader = torch.utils.data.DataLoader(
        instantiate(config.train_dataset),
        collate_fn=dataset.CityscapesDataset.collate_fn,
        **config.train_dataloader,
    )
    test_dataloader = torch.utils.data.DataLoader(
        instantiate(config.test_dataset),
        collate_fn=dataset.CityscapesDataset.collate_fn,
        **config.test_dataloader,
    )

    features = collect(train_dataloader, encoder, device)
    centroids = cluster(features, config.n_classes)

    for (x, labels, _, bounds) in test_dataloader:
        x = x.to(device)
        labels = labels.to(device)
        bounds = bounds.to(device)

        features = encode(labels, centroids, config.n_features, device)
        x_fake = generator(torch.cat((labels, bounds, features), dim=1))

        show_tensor_images(x_fake.to(x.dtype))
        show_tensor_images(x)
