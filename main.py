import random
from collections import deque

#import gym
import numpy as np
import tensorflow as tf

from env import *

ENV_NAME = "YUSEONG"
SOLVED_REWARD = 200
DONE_REWARD = 195
MAX_EPISODES = 100
DISCOUNT = 0.99
HIDDEN_COUNT = 64
MEMORY_LEN = 20

ACTION = [[0,29],[1,28],[2,27],[3,26],[4,25],[5,24],[6,23],[7,22],[8,21],[9,20],[10,19],[11,18],[12,17],[13,16],[14,15],[15,14],[16,13],[17,12],[18,11],[19,10],[20,9],[21,8],[22,7],[23,6],[24,5],[25,4],[26,3],[27,2],[28,1],[29,0]]

class Agent:
    """Agent implements the advantage actor critic (a2c) reinforcement learning model."""

    def __init__(self, env):
        # Inputs to the agent; only states is needed for picking the action, the rest are needed for training.
        #self.states = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=tf.float32)
        #self.next_states = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=tf.float32)
        self.states = tf.placeholder(shape = [None, 10], dtype=tf.float32)
        self.next_states = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        self.actions = tf.placeholder(dtype=tf.int64)
        self.rewards = tf.placeholder(dtype=tf.float32)
        self.solved = tf.placeholder(dtype=tf.bool)

        # Critic predicts the sum of the current and discounted future rewards, of the current and the next state.
        hidden_params = {
            "units": HIDDEN_COUNT,
            "activation": tf.nn.relu,
            "kernel_initializer": tf.glorot_uniform_initializer(),
        }
        batch_normalization = tf.layers.BatchNormalization()
        hidden1 = tf.layers.Dense(**hidden_params)
        hidden2 = tf.layers.Dense(**hidden_params)
        output = tf.layers.Dense(units=1, kernel_initializer=tf.ones_initializer())
        critic_nn = lambda x: output(hidden2(hidden1(batch_normalization(x))))
        critic = critic_nn(self.states)
        next_critic = critic_nn(self.next_states)  # Shares weights with critic.
        # Iff the episode was solved, i.e. it reached the maximum number of steps, the last step has future value.
        next_critic = tf.cond(self.solved, lambda: next_critic, lambda: tf.concat([next_critic[:-1], [[0]]], axis=0))
        next_critic = tf.stop_gradient(DISCOUNT * next_critic)  # Only critic is trained.
        critic_loss = tf.losses.mean_squared_error(labels=self.rewards + next_critic, predictions=critic)

        # Actor maintains a policy, i.e. probabilities for choosing actions.
        action_count =  30
        actor = tf.layers.batch_normalization(self.states)
        for _ in range(2):
            actor = tf.layers.dense(actor, **hidden_params)
        actor_logits = tf.layers.dense(actor, action_count, kernel_initializer=tf.ones_initializer())
        advantage = self.rewards + next_critic - tf.stop_gradient(critic)
        # Softmax cross entropy = -sum y log y'; y i.e. actions here is 1 only for the action being trained, and
        # log y' * advantage is the policy gradient (y' is the policy calculated by the network).
        actor_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.actions, logits=actor_logits) * advantage

        # Optimizers train the agent; actor_action picks the action to perform based on the current policy.
        self.critic_optimizer = tf.train.AdamOptimizer().minimize(critic_loss)
        self.actor_optimizer = tf.train.AdamOptimizer().minimize(actor_loss)
        self.actor_action = tf.squeeze(tf.random.multinomial(actor_logits, 1))

    def get_action(self, sess, state):
        """get_action returns the action (0 or 1) the agent wants to take in the current state."""
        return sess.run(self.actor_action, feed_dict={self.states: [state]})

    def train(self, sess, episode, solved):
        """train trains the agent with one episode."""
        sess.run(
            [self.critic_optimizer, self.actor_optimizer],
            feed_dict={
                self.states: [e["state"] for e in episode],
                self.actions: [e["action"] for e in episode],
                self.next_states: [e["next_state"] for e in episode],
                self.rewards: [e["reward"] for e in episode],
                self.solved: solved,
            },
        )


def run():
    """run runs the environment and uses the agent to balance the cartpole."""


    env = YuseongEnv()

    agent = Agent(env)
    past100 = np.zeros(100)
    past10 = np.zeros(10)
    memory = deque(maxlen=MEMORY_LEN)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i_episode in range(MAX_EPISODES):
            count = 0
            print("before")
            state, if_pre = env.reset(i_episode)
            print(f'{i_episode} epsiode pre-train')
            episode = []
            for k in range(100):
                # env.render()
                count +=1
                action = agent.get_action(sess, state)
                next_state, reward, done = env.step(ACTION[action], if_pre)
                print(f'{i_episode} epsiode {count} th train')
                episode.append({"action": action, "state": state, "next_state": next_state, "reward": reward})
                state = next_state

                if done:
                    break
            env.stop()
            total_reward = sum(e["reward"] for e in episode)

            past100[i_episode % 100] = total_reward
            if i_episode >= 100 and past100.mean() >= DONE_REWARD:
                print("done at", i_episode, "with past 100 avg reward", past100.mean())
                break

            past10[i_episode % 10] = total_reward
            if i_episode >= 10 and i_episode % 10 == 0:
                print("episode:", i_episode, "last 10; min:", past10.min(), "avg:", past10.mean(), "max:", past10.max())

            solved = total_reward >= SOLVED_REWARD
            agent.train(sess, episode, solved)
            memory.append((episode, solved))
            for episode, solved in random.sample(memory, min(len(memory), MEMORY_LEN // 2)):
                agent.train(sess, episode, solved)


if __name__ == "__main__":
    run()