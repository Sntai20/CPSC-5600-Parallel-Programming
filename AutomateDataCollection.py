import argparse
import subprocess
import math

add_counts = [ 2, 3, 4, 5, 6 ]
mul_counts = [ 100000, 200000, 300000, 400000, 500000 ]

sample_count = 10

# Add nosample switch
def main():
    parser = argparse.ArgumentParser(description="A script with a nosample switch")
    parser.add_argument('--nosample', action='store_true', help='Disable sampling')
    parser.add_argument('project_name', type=str, help='Input project name string where the program is in')
    parser.add_argument('program_name', type=str, help='Input program name string to process')
    parser.add_argument('--mpi', action='store_true', help='Use mpiexec')
    
    args = parser.parse_args()
    print(f"Project name: {args.project_name}")
    print(f"Program name: {args.program_name}")
    if args.nosample:
        print("Sampling is disabled.")
        for i in add_counts:
            for j in mul_counts:
                minimum = math.inf
                for k in range(sample_count):
                    cmd = [ f"{args.project_name}/out/{args.program_name}", str(i), str(j) ]
                    output_str = subprocess \
                        .run(cmd,stdout=subprocess.PIPE) \
                        .stdout.decode('utf-8')
                    # Ignore if output_str is empty.
                    if (output_str != ''):
                        time = float(output_str)
                        minimum = min(minimum,time)
                print(f"{minimum},",end="")
            print()
    else:
        print("Sampling is enabled.")
        for i in add_counts:
            for j in mul_counts:
                if args.mpi:
                    cmd = ["/usr/lib64/openmpi/bin/mpiexec", "-np", "20", f"{args.project_name}/out/{args.program_name}", str(i), str(j)]
                else:
                    cmd = [ f"{args.project_name}/out/{args.program_name}", str(i), str(j) ]
                
                output_str = subprocess \
                    .run(cmd,stdout=subprocess.PIPE) \
                    .stdout.decode('utf-8')

                # Ignore if output_str is empty.
                if (output_str != ''):
                    time = float(output_str)
                    print(f"{time},",end="")
            print()

if __name__ == "__main__":
    main()