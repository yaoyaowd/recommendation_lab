WORK_DIR = '/Users/dwang/irgan/item_recommendation/ml-100k/'
INPUT = '271479131.tsv'
TRAIN = 'wish-train.tsv'
TEST = 'wish-test.tsv'

from random import shuffle

line_cnt = 0
user_map = {}
pd_map = {}
appeared = {}
data = []
with open(WORK_DIR + INPUT, 'r') as input:
    for l in input:
        line_cnt += 1
        if line_cnt == 1:
            continue
        items = l.strip().split('\t')
        if items[0] not in user_map:
            user_map[items[0]] = len(user_map)
        if items[1] not in pd_map:
            pd_map[items[1]] = len(pd_map)
        key = items[0] + '\t' + items[1]
        if key in appeared:
            continue
        appeared[key] = 1
        data.append((user_map[items[0]], pd_map[items[1]]))

print 'total users: ', len(user_map)
print 'total products: ', len(pd_map)

shuffle(data)
with open(WORK_DIR + TRAIN, 'w') as output:
    for i in data[:len(data) / 2]:
        output.write(str(i[0]) + '\t' + str(i[1]) + '\t5\n')

with open(WORK_DIR + TEST, 'w') as output:
    for i in data[len(data) / 2 + 1:]:
        output.write(str(i[0]) + '\t' + str(i[1]) + '\t5\n')