import collections
import os
import pickle
import numpy as np
from pyflann import FLANN
# Catuious in using my own FLANN version; you need to keep the original embedding vector!

# ngtpy is buggy. (incremental remove and add is fragile)
#import ngtpy

class FastDictionary(object):
    def __init__(self,
                 maxlen,
                 seed=0,
                 cores=4,
                 trees=1):
        self.flann = FLANN(
            algorithm='kdtree',
            random_seed=seed,
            cores=cores,
            trees=trees,
        )

        self.counter = 0

        self.contents_lookup = {} #{oid: (e,q)}
        self.p_queue = collections.deque() #priority queue contains; list of (priotiry_value,oid)
        self.maxlen = maxlen

    def save(self,dir,fname,it=None):
        fname = f'{fname}' if it is None else f'{fname}-{it}'

        with open(os.path.join(dir,fname),'wb') as f:
            pickle.dump((self.contents_lookup,self.p_queue,self.maxlen),f)

    def restore(self,fname):
        with open(fname,'rb') as f:
            _contents_lookup, _p_queue, maxlen = pickle.load(f)

            assert self.maxlen == maxlen, (self.maxlen,maxlen)

        new_oid_lookup = {}
        E,Q = [],[]
        for oid,(e,q) in _contents_lookup.items():
            E.append(e)
            Q.append(q)

            new_oid, self.counter = self.counter, self.counter+1
            new_oid_lookup[oid] = new_oid

        E = np.array(E)

        # Rebuild KD-Tree
        self.flann.build_index(E)

        # Reallocate contents_lookup
        for new_oid,(e,q) in enumerate(zip(E,Q)):
            assert e.base is E
            self.contents_lookup[new_oid] = (e,q)

        # Rebuild Heap
        while len(_p_queue) > 0:
            oid = _p_queue.popleft()

            if not oid in new_oid_lookup:
                continue
            self.p_queue.append(new_oid_lookup[oid])


    def add(self,E,Contents):
        assert not np.isnan(E).any(), ('NaN Detected in Add',np.argwhere(np.isnan(E)))
        assert len(E) == len(Contents)
        assert E.ndim == 2 and E.shape[1] == 64, E.shape

        if self.counter == 0:
            self.flann.build_index(E)
        else:
            self.flann.add_points(E)
        Oid, self.counter = np.arange(self.counter,self.counter+len(E),dtype=np.uint32), self.counter + len(E)

        for oid,e,content in zip(Oid,E,Contents):
            assert e.base is E or e.base is E.base

            self.contents_lookup[oid] = (e,content)
            self.p_queue.append(oid)

            if len(self.contents_lookup) > self.maxlen:
                while not self.p_queue[0] in self.contents_lookup:
                    self.p_queue.popleft() #invalidated items due to update, so just pop.

                old_oid = self.p_queue.popleft()

                ret = self.flann.remove_point(old_oid)
                if ret <= 0:
                    raise Exception(f'remove point error {ret}')
                del self.contents_lookup[old_oid]

    def update(self,Oid,E,Contents):
        """
        Basically, same this is remove & add.
        This code only manages a heap more effectively; since delete an item in the middle of heap is not trivial!)
        """
        assert not np.isnan(E).any(), ('NaN Detected in Updating',np.argwhere(np.isnan(E)))
        assert len(np.unique(Oid)) == len(Oid)
        assert E.ndim == 2 and E.shape[1] == 64, E.shape

        # add new Embeddings
        self.flann.add_points(E)
        NewOid, self.counter = np.arange(self.counter,self.counter+len(E),dtype=np.uint32), self.counter + len(E)

        for oid,new_oid,e,content in zip(Oid,NewOid,E,Contents):
            assert e.base is E or e.base is E.base

            self.contents_lookup[new_oid] = (e,content)
            self.p_queue.append(new_oid)

            # delete from kd-tree
            ret = self.flann.remove_point(oid)
            if ret <= 0:
                raise Exception(f'remove point error {ret}')
            # delete from contents_lookup
            del self.contents_lookup[oid]
            # I cannot remove from p_queue, but it will be handeled in add op.

    def query_knn(self,E,K=100):
        assert not np.isnan(E).any(), ('NaN Detected in Querying',np.argwhere(np.isnan(E)))

        flatten = False
        if E.ndim == 1:
            E = E[None]
            flatten = True

        Oids, Dists, C = self.flann.nn_index(E,num_neighbors=K)

        if C != len(E)*K:
            print(f'Not enough neighbors ({np.count_nonzero(Dists>=0.)} == {C}) != {len(E)}*{K}, rebuild and try again...')
            self.flann.rebuild_index()
            Oids, Dists, C = self.flann.nn_index(E,num_neighbors=K)

        # TODO: Hmm. Dists sometimes becomes NaN
        #assert np.count_nonzero(np.isnan(Dists)) == 0, 'pyflann returned a NaN for a distance'
        if np.count_nonzero(np.isnan(Dists)) > 0:
            print('warning: NaN Returned as a distance')
            Dists = np.nan_to_num(Dists,copy=False)

        NN_E = np.zeros((len(E),K,E.shape[1]),np.float32)
        NN_Q = np.zeros((len(E),K),np.float32)
        Len = np.count_nonzero(Dists>=0.,axis=1)

        assert np.sum(Len) == C, f'{np.sum(Len)} != {C}'
        assert C > 0, 'Nothing returned...'

        for b,oids in enumerate(Oids):
            for k,oid in enumerate(oids[:Len[b]]): #drop if not enough NN retrieved.
                e,q = self.contents_lookup[oid]

                NN_E[b,k] = e
                NN_Q[b,k] = q

        if flatten:
            return Oids[0][:Len[0]], NN_E[0][:Len[0]], NN_Q[0][:Len[0]]
        else:
            return Oids, NN_E, NN_Q, Len

if __name__ == "__main__":
    pass
