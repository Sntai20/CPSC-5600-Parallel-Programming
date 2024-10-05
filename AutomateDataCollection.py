import argparse
import subprocess
import math

add_counts = [ 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 ]
mul_counts = [ 1, 2, 3, 4, 5, 6, 7, 8 ]

sample_count = 10

# Add nosample switch
def main():
    parser = argparse.ArgumentParser(description="A script with a nosample switch")
    parser.add_argument('--nosample', action='store_true', help='Disable sampling')
    parser.add_argument('project_name', type=str, help='Input project name string where the program is in')
    parser.add_argument('program_name', type=str, help='Input program name string to process')
    
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