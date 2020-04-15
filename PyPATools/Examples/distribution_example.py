from PyPATools.particles import ParticleDistribution
import time

ntests = 10

pd = ParticleDistribution()

start_time = time.time()
for i in range(ntests):
    pd.generate_random_sample(10000000)

print("Average of {} loops was {} s.".format(ntests, (time.time() - start_time) / ntests))
