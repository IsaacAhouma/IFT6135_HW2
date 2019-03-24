import numpy as np
import os
import matplotlib.pyplot as plt
cwd = os.getcwd()
print(cwd)

epochs = range(40)
results_path = 'RNN_SGD_model=RNN_optimizer=SGD_initial_lr=0.0001_batch_size=2'
results = np.load(results_path + '/learning_curves.npy')[()]
valid_ppl = results[('val_ppls')]
train_ppl = results[('train_ppls')]

# Extract wall clock time from log file
logs_txt = open(results_path+'/log.txt')
i = 0
wall_clock_time = []
running_time = 0
for line in logs_txt:
    _, time = line.split('time (s) spent in epoch: ')
    i += 1
    running_time += float(time.strip())
    wall_clock_time.append(running_time)

fig = plt.figure()
ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)

ax.plot(epochs, train_ppl, 'b', epochs, valid_ppl, 'r')
ax.legend(['train', 'valid'])
ax.set_xlabel("Epochs", color="k")
ax.set_ylabel("Perplexity", color="k")
ax.tick_params(axis='x', colors="k")
ax.tick_params(axis='y', colors="k")
ax2.plot(wall_clock_time, train_ppl, "b", wall_clock_time, valid_ppl, 'r')
ax2.xaxis.tick_top()
ax2.set_xlabel('Wall Clock Time (s)', color="k")
ax2.xaxis.set_label_position('top')

plt.savefig(results_path + '/learning_curves.jpg')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()