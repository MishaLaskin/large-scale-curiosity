# Notes on saving / loading the policy

In `run.py` agent is specified at PpoOptimizer from the `cppo_agent.py` file. The policy is set as agent.policy so this is what we want to save


 
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./tmp/model.ckpt-0.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))