import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('log_file', help='log file')
parser.add_argument('stat_names', help='comma-separated stat names')
args = parser.parse_args()

stat_names = args.stat_names.split(',')

iter_nums = []
statistics = {e: [] for e in stat_names}
with open(args.log_file, encoding='utf-8') as f:
    for line in f:
        m1 = re.search(r'iter: (\d+)', line)
        if m1:
            iter_num = int(m1.group(1))
            iter_nums.append(iter_num)
            for stat_name in stat_names:
                m2 = re.search(stat_name+r': (\S+)', line)
                assert m2, 'can\'t find stat \'{}\' in this line:\n{}'.format(stat_name, line)
                value = float(m2.group(1))
                statistics[stat_name].append(value)

fig, axs = plt.subplots(len(stat_names), sharex=True, squeeze=False)
fig.suptitle('training progress')
for i, (stat_name, stats) in enumerate(statistics.items()):
    axs[i][0].set_title(stat_name)
    axs[i][0].plot(iter_nums, stats)
plt.show()
