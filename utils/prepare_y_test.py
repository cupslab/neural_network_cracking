import pwd_guess as pg
import string
import random
import time
import time

config = pg.ModelDefaults(
    rare_character_optimization=True,
    uppercase_character_optimization=True,
    intermediate_fname = 'test.sqlite')
config.set_intermediate_info('rare_character_bag', '~!`@#$%^&*()-_=+[]{};:<>.')
f = pg.FuzzyTrieTrainer(None, config)

def random_string(slen):
    return ''.join([random.choice(config.char_bag) for _ in range(slen)])

# randomdata = []
# for _ in range(1000):
#     tuples = []
#     for _ in range(random.randint(1, 10)):
#         tuples.append((random.choice(config.char_bag), random.randint(5, 10)))
#     randomdata.append(tuples)

# print('starting')

# start = time.clock()
# for _ in range(1000):
#     f.prepare_y_data(randomdata)
# end = time.clock()

# print("%.2gs" % (end - start))


data = [random_string(random.randint(1, 10)) for _ in range(1000)]

print('starting')

start = time.clock()
for _ in range(1000):
    f.ctable.encode_many(data)
end = time.clock()

print("%.2gs" % (end - start))
