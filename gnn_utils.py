import dgl
import numpy as np
import torch as th
import torchvision.transforms as transforms

def aux_transforms(img_batch, trans_list, replica):
    img_batch = img_batch.unsqueeze(1).repeat(1, replica, 1, 1, 1)
    idx, emb = [], []
    for i, img in enumerate(img_batch):
        emb.append(img[0].unsqueeze(0))
        idx.extend([i])
        for trans in trans_list:
            if trans == 'crop':
                transform = transforms.RandomResizedCrop(size=(224, 224))
            elif trans == 'horflip':
                transform = transforms.RandomHorizontalFlip()
            else:
                raise Exception(f"{trans} not implemented")
            trans_img = transform(img)
            emb.append(trans_img)
            idx.extend([i] * len(trans_img))
    emb = th.cat(emb, dim=0)
    return emb, th.tensor(idx)

def build_g(emb, idx):
    g_list = []
    for i in th.unique(idx):
        _idx = th.where(idx == i)[0]
        g = dgl.graph(
            (
                th.tensor([0] * len(_idx)),
                th.tensor(list(range(len(_idx))))
            )
        )
        g = dgl.to_bidirected(g).to(emb.device)
        g.ndata['feat'] = emb[_idx]
        g_list.append(g)
    g = dgl.batch(g_list)
    return g

if __name__ == '__main__':
    emb, idx = aux_transforms(th.ones((2, 3, 224, 224)), trans_list=['crop', 'horflip'], replica=2)
    g = build_g(emb, idx)
    print(1)

