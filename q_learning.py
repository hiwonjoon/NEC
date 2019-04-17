import os
import itertools
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm

from commons.ops import *
from fast_dictionary import FastDictionary
from replay_buffer import ReplayBuffer
from libs.atari_wrappers import make_atari, wrap_deepmind

def _build(net,x):
    for block in net: x = block(x)
    return x

class NEC(object):
    clip_grad_norm = 10.0
    lr = 1e-4

    def __init__(self,
                 num_ac,
                 memory_max_len,
                 K,
                 embed_len,
                 delta=1e-5,
                 q_lr=0.01,
                 ):
        self.delta = delta
        self.q_lr = q_lr
        self.K = K
        self.embed_len = embed_len

        self.Qa = [FastDictionary(memory_max_len) for _ in range(num_ac)]

        # Experience reaclled from a replay buffer through FastDictionary
        self.s = tf.placeholder(tf.float32,[None,84,84,4]) #[B,seq_len,state_dims]

        self.nn_es = tf.placeholder(tf.float32,[None,None,embed_len]) #[B,K,embed_len]
        self.nn_qs = tf.placeholder(tf.float32,[None,None])
        self.nn_len = tf.placeholder(tf.int32,[None])
        self.target_q = tf.placeholder(tf.float32,[None])

        with tf.variable_scope('weights') as param_scope:
            self.param_scope = param_scope

            self.net = [
                Conv2d('c1',4,32,k_h=8,k_w=8,d_h=4,d_w=4,padding='VALID',data_format='NHWC'),
                lambda x: tf.nn.relu(x),
                Conv2d('c2',32,64,k_h=4,k_w=4,d_h=2,d_w=2,padding='VALID',data_format='NHWC'),
                lambda x: tf.nn.relu(x),
                Conv2d('c3',64,64,k_h=3,k_w=3,d_h=1,d_w=1,padding='VALID',data_format='NHWC'),
                lambda x: tf.nn.relu(x),
                Linear('fc1',3136,512),
                lambda x: tf.nn.relu(x),
                Linear('fc2',512,embed_len)
            ]

        self.embed = _build(self.net,self.s)

        dists = tf.reduce_sum((self.nn_es - self.embed[:,None])**2,axis=2) +self.delta #[B,K]
        mask = tf.sequence_mask(self.nn_len,maxlen=self.K,dtype=tf.float32) #[B,K]
        kernel = mask * (1. / dists) #[B,K]

        self.q = tf.reduce_sum(kernel * self.nn_qs, axis=1) / tf.reduce_sum(kernel,axis=1)

        self.loss = tf.reduce_mean((self.target_q - self.q)**2)

        # Optimize op
        #self.optim = tf.train.AdamOptimizer(1e-4)
        self.optim = tf.train.RMSPropOptimizer(self.lr)
        self.update_op = self.optim.minimize(self.loss,var_list=self.parameters(train=True))

        self.nn_es_gradient, self.nn_qs_gradient = tf.gradients(self.loss, [self.nn_es,self.nn_qs])

        self.new_nn_es = self.nn_es - self.lr * tf.clip_by_norm(self.nn_es_gradient,self.clip_grad_norm,axes=[2])
        self.new_nn_qs = self.nn_qs - self.q_lr * tf.clip_by_value(self.nn_qs_gradient,-self.clip_grad_norm,self.clip_grad_norm)

        self.saver = tf.train.Saver(var_list=self.parameters(train=True),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def save(self,dir,it=None):
        sess = tf.get_default_session()
        self.saver.save(sess,dir+'/model.ckpt',global_step=it,write_meta_graph=False)

        for i,Q in enumerate(self.Qa):
            Q.save(dir,f'Q-{i}.pkl',it)

    def restore(self,model_file):
        model_dir,file_name = os.path.split(model_file)
        it = None if not '-' in file_name else int(file_name.split('-')[-1])

        sess = tf.get_default_session()

        self.saver.restore(sess,model_file)

        for i,Q in enumerate(self.Qa):
            memory_name = f'Q-{i}.pkl' if it is None else f'Q-{i}.pkl-{it}'

            Q.restore(os.path.join(model_dir,memory_name))

    # NEC Impl
    def _embed(self,b_s,max_batch_size=1024):
        sess = tf.get_default_session()

        b_e = []
        for i in range(0,len(b_s),max_batch_size):
            b_e.append(
                sess.run(self.embed,feed_dict={self.s:b_s[i:i+max_batch_size]}))
        return np.concatenate(b_e,axis=0)

    def _read_table(self,e,Q,K):
        oids, nn_es, nn_qs = Q.query_knn(e,K=K)

        dists = np.sum((nn_es - e)**2,axis=1)
        kernel = 1 / (dists + self.delta)

        q = np.sum(kernel * nn_qs) / np.sum(kernel)

        return oids, q

    def policy(self,s):
        e = self._embed(s[None])[0]

        qs = [self._read_table(e,Q,K=self.K)[1] for Q in self.Qa]

        ac = np.argmax(qs)
        return ac, (e,qs[ac])

    def append(self,e,a,q):
        sess = tf.get_default_session()

        self.Qa[a].add(e[None],[(e,q)])

    def update(self,b_s,b_a,b_q):
        sess = tf.get_default_session()

        b_e = self._embed(b_s)

        b_nn_es = np.zeros((len(b_s),self.K,self.embed_len),np.float32)
        b_nn_qs = np.zeros((len(b_s),self.K),np.float32)
        b_nn_len = np.zeros((len(b_s),),np.int32)

        # for debugging
        #_new_b_nn_es = np.zeros((len(b_s),self.K,self.embed_len),np.float32)
        #_new_b_nn_qs = np.zeros((len(b_s),self.K),np.float32)

        for a,Q in enumerate(self.Qa):
            idxes = np.where(b_a==a)
            if len(idxes[0]) == 0:
                continue

            Oids, nn_Es, nn_Qs, nn_Len = Q.query_knn(b_e[idxes],K=self.K)

            b_nn_es[idxes] = nn_Es
            b_nn_qs[idxes] = nn_Qs
            b_nn_len[idxes] = nn_Len

            # Update the table (embedding & q itself)
            new_Es, new_Qs = \
                sess.run([self.new_nn_es,self.new_nn_qs],feed_dict={
                    self.s:b_s[idxes],
                    self.nn_es:nn_Es,
                    self.nn_qs:nn_Qs,
                    self.nn_len:nn_Len,
                    self.target_q:b_q[idxes],
                })

            #_new_b_nn_es[idxes] = new_Es
            #_new_b_nn_qs[idxes] = new_Qs

            total_len = np.sum(nn_Len)

            oids = np.zeros((total_len,),np.uint32)
            new_nn_es = np.zeros((total_len,self.embed_len),np.float32)
            new_nn_qs = np.zeros((total_len,),np.float32)

            i = 0
            for b,l in enumerate(nn_Len):
                oids[i:i+l] = Oids[b,:l]
                new_nn_es[i:i+l] = nn_Es[b,:l]
                new_nn_qs[i:i+l] = nn_Qs[b,:l]
                i += l

            _, unique_idxes = np.unique(oids,return_index=True)
            Q.update(oids[unique_idxes],
                     new_nn_es[unique_idxes],
                     list(zip(new_nn_es[unique_idxes],new_nn_qs[unique_idxes])))

        # Update the embedding network
        #before_q, loss, _ = sess.run([self.q,self.loss,self.update_op],feed_dict={
        before_q, loss = sess.run([self.q,self.loss],feed_dict={
            self.s:b_s,
            self.nn_es:b_nn_es,
            self.nn_qs:b_nn_qs,
            self.nn_len:b_nn_len,
            self.target_q:b_q,
        })

        #new_q = sess.run(self.q,feed_dict={
        #    self.s:b_s,
        #    self.nn_es:_new_b_nn_es,
        #    self.nn_qs:_new_b_nn_qs,
        #})

        return loss


def train(
    args,
    log_dir,
    seed,
    env_id,
    replay_buffer_len,
    memory_len,
    p,             # #nn items; reported number is 50
    embed_size,    # embedding vector length; reported number is ?
    gamma,         # discount value; reported number is 0.99
    N,             # N-step bootstrapping; reported number is 100
    update_period, # the reported number is 16
    batch_size,    # the reported number is 32
    init_eps,
    **kwargs
):
    # another hyper params
    _gw = np.array([gamma**i for i in range(N)])
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.99 #explonential decaying factor

    # expr setting
    Path(log_dir).mkdir(parents=True,exist_ok='temp' in log_dir)

    with open(os.path.join(log_dir,'args.txt'),'w') as f:
        f.write( str(args) )

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    # Env
    env = wrap_deepmind(make_atari(env_id),
                        episode_life=False,
                        clip_rewards=False,
                        frame_stack=True,
                        scale=False)
    num_ac = env.action_space.n

    # ReplayBuffer
    replay_buffer = ReplayBuffer(replay_buffer_len)

    # Neural Episodic Controller
    nec = NEC(num_ac,memory_len,p,embed_size)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(os.path.join(log_dir,'tensorboard'))
    def _write_scalar(it,it_type,tag,value):
        summary = tf.Summary(value=[tf.Summary.Value(tag=f"{tag}/{it_type}", simple_value=value)])
        summary_writer.add_summary(summary,global_step=it)

    ####### Setup Done

    num_frames = 0 # #frames observed
    num_updates = 0

    # Fill up the memory and replay buffer with a random policy
    for ep in range(init_eps):
        ob = env.reset()

        obs,acs,rewards = [ob],[],[]
        for _ in itertools.count():
            ac = np.random.randint(num_ac)

            ob,r,done,_ = env.step(ac)

            obs.append(ob)
            acs.append(ac)
            rewards.append(r)

            num_frames += 1

            if done:
                break

        Rs = [np.sum(_gw[:len(rewards[i:i+N])]*rewards[i:i+N]) for i in range(len(rewards))]

        obs = np.array(obs)
        es = nec._embed(obs)

        for ob,e,a,R in zip(obs,es,acs,Rs):
            nec.append(e, a, R)

            replay_buffer.append(ob,a,R)

    # Training!
    try:
        for ep in itertools.count():
            ob = env.reset()

            obs,acs,rewards,es,Vs = [ob],[],[],[],[]
            for t in itertools.count():
                # Epsilon Greedy Policy
                ac, (e,V) = nec.policy(ob)
                if np.random.random() < epsilon:
                    ac = np.random.randint(num_ac)

                ob,r,done,_ = env.step(ac)

                obs.append(ob)
                acs.append(ac)
                rewards.append(r)
                es.append(e)
                Vs.append(V)

                num_frames += 1

                # Train on random minibatch from replacy buffer
                if num_frames % update_period == 0:
                    b_s,b_a,b_R = replay_buffer.sample(batch_size)
                    loss = nec.update(b_s,b_a,b_R)

                    num_updates += 1

                    if num_updates % 100 == 0:
                        print(f'[{num_frames}/{num_updates}] loss: {loss}')

                    _write_scalar(it=num_frames,it_type='per_frames',tag='loss',value=loss)
                    _write_scalar(it=num_updates,it_type='per_updates',tag='loss',value=loss)
                    _write_scalar(it=num_frames,it_type='per_frames',tag='num_updates',value=num_updates)

                if t >= N:
                    # N-Step Bootstrapping
                    # TODO: implement the efficient version
                    R = np.sum(_gw * rewards[t-N:t]) + gamma**N*Vs[t] #R_{t-N}

                    # append to memory
                    nec.append(es[t-N], acs[t-N], R)

                    # append to replay buffer
                    replay_buffer.append(obs[t-N], acs[t-N], R)

                if done:
                    break

            print(f'Episode {ep} -- Ep Len: {len(obs)} Acc Reward: {np.sum(rewards)} current epsilon: {epsilon}')
            _write_scalar(tag='ep',value=ep,it=num_frames,it_type='per_frames')
            _write_scalar(tag='ep_len',value=len(obs),it=num_frames,it_type='per_frames')
            _write_scalar(tag='ep_len',value=len(obs),it=ep,it_type='per_episode')
            _write_scalar(tag='eps_reward',value=np.sum(rewards),it=num_frames,it_type='per_frames')
            _write_scalar(tag='eps_reward',value=np.sum(rewards),it=ep,it_type='per_episode')
            _write_scalar(tag='epsilon',value=epsilon,it=ep,it_type='per_episode')

            # Remaining items which is not bootstrappable; partial trajectory close to end of episode
            # Append to memory & replay buffer
            for t in range(len(rewards)-N,len(rewards)):
                R = np.sum([gamma**(i-t)*rewards[i] for i in range(t,len(rewards))])
                nec.append(es[t], acs[t], R)
                replay_buffer.append(obs[t], acs[t], R)

            # epsilon decay
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

    except KeyboardInterrupt:
        nec.save(log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # expr setting
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--mode',default='train',choices=['train'])
    # Env
    parser.add_argument('--env_id', default='PongNoFrameskip-v4', help='Select the domain name; eg) cartpole')
    parser.add_argument('--gamma',type=float,default=0.99)
    # network
    parser.add_argument('--embed_size',type=int,default=64)
    parser.add_argument('--memory_len',type=int,default=int(5*1e5))
    parser.add_argument('--replay_buffer_len',type=int,default=int(1e5))
    parser.add_argument('--p',type=int,default=50)
    # Training
    parser.add_argument('--init_eps',type=int,default=10,help='# episodes with random policy for initialize memory and replay buffer')
    parser.add_argument('--N',type=int,default=100,help='N-step-bootstrapping')
    parser.add_argument('--update_period',type=int,default=16)
    parser.add_argument('--batch_size',type=int,default=32)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args=args,**vars(args))


