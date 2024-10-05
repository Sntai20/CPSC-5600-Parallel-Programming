import argparse
import subprocess
import math

# add_counts = [ 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 ]
# mul_counts = [ 1, 2, 3, 4, 5, 6, 7, 8 ]
add_counts = [ 10, 100, 1000, 10000, 100000 ]
mul_counts = [ 1, 2, 3, 4, 5 ]

sample_count = 10

# Add nosample switch
def main():
    parser = argparse.ArgumentParser(description="A script with a nosample switch")
    parser.add_argument('--nosample', action='store_true', help='Disable sampling')
    parser.add_argument('input_string', type=str, help='Input string to process')
    
    args = parser.parse_args()
    
    if args.nosample:
        print("Sampling is disabled.")
        print(f"Program name: {args.input_string}")
        for i in add_counts:
            for j in mul_counts:
                minimum = math.inf
                for k in range(sample_count):
                    cmd = [ f"out/{args.input_string}", str(i), str(j) ]
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
        print(f"Program name: {args.input_string}")
        for i in add_counts:
            for j in mul_counts:
                cmd = [ f"out/{args.input_string}", str(i), str(j) ]
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