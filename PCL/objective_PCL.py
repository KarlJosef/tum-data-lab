import tensorflow as tf
import numpy as np

class Objective(object):

    def __init__(self, learning_rate,
                clip_norm=5,
                policy_weight=1.0,
                critic_weight=0.1,
                tau=0.1,
                gamma=1.0,
                rollout=10,
                eps_lambda=0.0,
                clip_adv=None):

        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout
        self.clip_adv = clip_adv

        self.eps_lambda = tf.get_variable(  # TODO: need a better way
              'eps_lambda', [], initializer=tf.constant_initializer(eps_lambda))
        self.new_eps_lambda = tf.placeholder(tf.float32, [])
        self.assign_eps_lambda = self.eps_lambda.assign(
              0.95 * self.eps_lambda + 0.05 * self.new_eps_lambda)



    def get_optimizer(self, learning_rate):
        """Optimizer for gradient descent ops."""
        return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=2e-4)

    def training_ops(self, loss, learning_rate=None):
        """Gradient ops."""
        opt = self.get_optimizer(learning_rate)
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)

        if self.clip_norm:
            grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
            tf.summary.scalar('grad_global_norm', global_norm)

        return opt.apply_gradients(zip(grads, params))

    def get(self, rewards, pads, values, final_values,
          log_probs, target_log_probs):
        """Get objective calculations."""
        raise NotImplementedError()


def discounted_future_sum(values, discount, rollout):
    """Discounted future sum of time-major values."""
    discount_filter = tf.reshape(
        discount ** tf.range(float(rollout)), [-1, 1, 1])
    expanded_values = tf.concat(
        [values, tf.zeros([rollout - 1, tf.shape(values)[1]])], 0)

    conv_values = tf.transpose(tf.squeeze(tf.nn.conv1d(
        tf.expand_dims(tf.transpose(expanded_values), -1), discount_filter,
        stride=1, padding='VALID'), -1))

    return conv_values


def discounted_two_sided_sum(values, discount, rollout):
    """Discounted two-sided sum of time-major values."""
    roll = float(rollout)
    discount_filter = tf.reshape(
      discount ** tf.abs(tf.range(-roll + 1, roll)), [-1, 1, 1])
    expanded_values = tf.concat(
      [tf.zeros([rollout - 1, tf.shape(values)[1]]), values,
       tf.zeros([rollout - 1, tf.shape(values)[1]])], 0)

    conv_values = tf.transpose(tf.squeeze(tf.nn.conv1d(
      tf.expand_dims(tf.transpose(expanded_values), -1), discount_filter,
      stride=1, padding='VALID'), -1))

    return conv_values


def shift_values(values, discount, rollout, final_values=0.0):
    """Shift values up by some amount of time.
    Those values that shift from a value beyond the last value
    are calculated using final_values.
    """
    roll_range = tf.cumsum(tf.ones_like(values[:rollout, :]), 0,
                         exclusive=True, reverse=True)
    final_pad = tf.expand_dims(final_values, 0) * discount ** roll_range
    return tf.concat([discount ** rollout * values[rollout:, :],
                    final_pad], 0)


class PCL(Objective):
    """PCL implementation.
    Implements vanilla PCL, Unified PCL, and Trust PCL depending
    on provided inputs.
    """

    def get(self, rewards, pads, values, final_values,
          log_probs, target_log_probs):

        not_pad = 1 - pads
        batch_size = tf.shape(rewards)[1]

        rewards = not_pad * rewards
        value_estimates = not_pad * values
        log_probs = not_pad * sum(log_probs)
        target_log_probs = not_pad * tf.stop_gradient(sum(target_log_probs))
        relative_log_probs = not_pad * (log_probs - target_log_probs)

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                               log_probs], 0)
        #prev_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
        #                            prev_log_probs], 0)
        relative_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                                        relative_log_probs], 0)

        sum_rewards = discounted_future_sum(rewards, self.gamma, self.rollout)
        sum_log_probs = discounted_future_sum(log_probs, self.gamma, self.rollout)
        #sum_prev_log_probs = discounted_future_sum(prev_log_probs, self.gamma, self.rollout)
        sum_relative_log_probs = discounted_future_sum(
            relative_log_probs, self.gamma, self.rollout)
        last_values = shift_values(value_estimates, self.gamma, self.rollout,
                                   final_values)

        future_values = (
            - self.tau * sum_log_probs
            - self.eps_lambda * sum_relative_log_probs
            + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        if self.clip_adv:
          adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
        policy_loss = -adv * sum_log_probs
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            tf.reduce_sum(policy_loss * not_pad, 0))
        critic_loss = tf.reduce_mean(
            tf.reduce_sum(critic_loss * not_pad, 0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            tf.reduce_sum(not_pad * adv * (-baseline_values + future_values), 0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
        tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
        tf.summary.histogram('future_values', future_values)
        tf.summary.histogram('baseline_values', baseline_values)
        tf.summary.histogram('advantages', adv)
        tf.summary.scalar('avg_rewards',
                          tf.reduce_mean(tf.reduce_sum(rewards, 0)))
        tf.summary.scalar('policy_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('critic_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss))
        tf.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss,
                future_values[self.rollout - 1:, :],
                gradient_ops, tf.summary.merge_all())