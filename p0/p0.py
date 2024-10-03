import subprocess
import math

add_counts = [ 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 ]
mul_counts = [ 1, 2, 3, 4, 5, 6, 7, 8 ]

sample_count = 10

for i in add_counts:
    for j in mul_counts:
        minimum = math.inf
        for k in range(sample_count):
            cmd = [ "./p0", str(i), str(j) ]
            output_str = subprocess \
                .run(cmd,stdout=subprocess.PIPE) \
                .stdout.decode('utf-8')
            time = float(output_str)
            minimum = min(minimum,time)
        print(f"{minimum},",end="")
    print()
