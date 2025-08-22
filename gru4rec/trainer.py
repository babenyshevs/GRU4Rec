import time
import numpy as np
import torch
from torch import nn

from .data import SessionDataIterator
from .model import GRU4RecModel
from .optimizers import IndexedAdagradM


class GRU4Rec:
    def __init__(
        self,
        layers=[100],
        loss="cross-entropy",
        batch_size=64,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
        embedding=0,
        constrained_embedding=True,
        n_epochs=10,
        bpreg=1.0,
        elu_param=0.5,
        logq=0.0,
        device=torch.device("cuda:0"),
    ):
        self.device = device
        self.layers = layers
        self.loss = loss
        self.set_loss_function(loss)
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.logq = logq
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        if embedding == "layersize":
            self.embedding = self.layers[0]
        else:
            self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs

    def set_loss_function(self, loss):
        if loss == "cross-entropy":
            self.loss_function = self.xe_loss_with_softmax
        elif loss == "bpr-max":
            self.loss_function = self.bpr_max_loss_with_elu
        else:
            raise NotImplementedError

    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k, v in kvargs.items():
            if not hasattr(self, k):
                print("Unkown attribute: {}".format(k))
                raise NotImplementedError
            else:
                if type(v) == str and type(getattr(self, k)) == list:
                    v = [int(l) for l in v.split("/")]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == "True" or v == "1":
                        v = True
                    elif v == "False" or v == "0":
                        v = False
                    else:
                        print("Invalid value for boolean parameter: {}".format(v))
                        raise NotImplementedError
                if k == "embedding" and v == "layersize":
                    self.embedding = "layersize"
                setattr(self, k, type(getattr(self, k))(v))
                if k == "loss":
                    self.set_loss_function(self.loss)
                print(
                    "SET   {}{}TO   {}{}(type: {})".format(
                        k,
                        " " * (maxk_len - len(k) + 3),
                        getattr(self, k),
                        " " * (maxv_len - len(str(getattr(self, k))) + 3),
                        type(getattr(self, k)),
                    )
                )
        if self.embedding == "layersize":
            self.embedding = self.layers[0]
            print(
                "SET   {}{}TO   {}{}(type: {})".format(
                    "embedding",
                    " " * (maxk_len - len("embedding") + 3),
                    getattr(self, "embedding"),
                    " " * (maxv_len - len(str(getattr(self, "embedding"))) + 3),
                    type(getattr(self, "embedding")),
                )
            )

    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            O = O - self.logq * torch.log(
                torch.cat([self.P0[Y[:M]], self.P0[Y[M:]] ** self.sample_alpha])
            )
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X) + 1e-24))

    def softmax_neg(self, X):
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        return e_x / e_x.sum(dim=1, keepdim=True)

    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            O = nn.functional.elu(O, self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            -torch.log(
                torch.sum(torch.sigmoid(target_scores - O) * softmax_scores, dim=1)
                + 1e-24
            )
            + self.bpreg * torch.sum((O**2) * softmax_scores, dim=1)
        )

    def fit(
        self,
        data,
        sample_cache_max_size=10000000,
        compatibility_mode=True,
        item_key="ItemId",
        session_key="SessionId",
        time_key="Time",
    ):
        self.error_during_train = False
        self.data_iterator = SessionDataIterator(
            data,
            self.batch_size,
            n_sample=self.n_sample,
            sample_alpha=self.sample_alpha,
            sample_cache_max_size=sample_cache_max_size,
            item_key=item_key,
            session_key=session_key,
            time_key=time_key,
            session_order="time",
            device=self.device,
        )
        if self.logq and self.loss == "cross-entropy":
            pop = data.groupby(item_key).size()
            self.P0 = torch.tensor(
                pop[self.data_iterator.itemidmap.index.values],
                dtype=torch.float32,
                device=self.device,
            )
        model = GRU4RecModel(
            self.data_iterator.n_items,
            self.layers,
            self.dropout_p_embed,
            self.dropout_p_hidden,
            self.embedding,
            self.constrained_embedding,
        ).to(self.device)
        if compatibility_mode:
            model._reset_weights_to_compatibility_mode()
        self.model = model
        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)
        for epoch in range(self.n_epochs):
            t0 = time.time()
            H = []
            for i in range(len(self.layers)):
                H.append(
                    torch.zeros(
                        (self.batch_size, self.layers[i]),
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,
                    )
                )
            c = []
            cc = []
            n_valid = self.batch_size
            reset_hook = lambda n_valid, finished_mask, valid_mask: self._adjust_hidden(
                n_valid, finished_mask, valid_mask, H
            )
            for in_idx, out_idx in self.data_iterator(
                enable_neg_samples=(self.n_sample > 0), reset_hook=reset_hook
            ):
                for h in H:
                    h.detach_()
                self.model.zero_grad()
                R = self.model.forward(in_idx, H, out_idx, training=True)
                L = self.loss_function(R, out_idx, n_valid) / self.batch_size
                L.backward()
                opt.step()
                L = L.cpu().detach().numpy()
                c.append(L)
                cc.append(n_valid)
                if np.isnan(L):
                    print(str(epoch) + ": NaN error!")
                    self.error_during_train = True
                    return
            c = np.array(c)
            cc = np.array(cc)
            avgc = np.sum(c * cc) / np.sum(cc)
            if np.isnan(avgc):
                print("Epoch {}: NaN error!".format(str(epoch)))
                self.error_during_train = True
                return
            t1 = time.time()
            dt = t1 - t0
            print(
                "Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]".format(
                    epoch + 1, avgc, dt, len(c) / dt, np.sum(cc) / dt
                )
            )

    def _adjust_hidden(self, n_valid, finished_mask, valid_mask, H):
        if (self.n_sample == 0) and (n_valid < 2):
            return True
        with torch.no_grad():
            for i in range(len(self.layers)):
                H[i][finished_mask] = 0
        if n_valid < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return False

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if device == self.device:
            return
        if hasattr(self, "model"):
            self.model = self.model.to(device)
            self.model.eval()
        self.device = device
        if hasattr(self, "data_iterator"):
            self.data_iterator.device = device
            if hasattr(self.data_iterator, "sample_cache"):
                self.data_iterator.sample_cache.device = device
        pass

    def savemodel(self, path):
        torch.save(self, path)

    @classmethod
    def loadmodel(cls, path, device="cuda:0"):
        gru = torch.load(path, map_location=device)
        gru.device = torch.device(device)
        if hasattr(gru, "data_iterator"):
            gru.data_iterator.device = torch.device(device)
            if hasattr(gru.data_iterator, "sample_cache"):
                gru.data_iterator.sample_cache.device = torch.device(device)
        gru.model.eval()
        return gru
