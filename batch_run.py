import os
import sys
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_simulation(config_path):
    try:
        # for compatibility across OS
        config_path_arg = config_path.replace(os.sep, '/')
        cmd = [sys.executable, 'run.py', config_path_arg]
        print(f"Starting simulation for {config_path}...")

        # Run the process
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully finished {config_path}")
            # Optional: print stdout if needed, or just keep it quiet
            # print(result.stdout)
        else:
            print(f"Error running {config_path}:\n{result.stderr}")
            print(f"Output:\n{result.stdout}")

    except Exception as e:
        print(f"Exception while running {config_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_run.py folder_containing_configs")
        sys.exit(1)

    config_folder = sys.argv[1]

    if not os.path.isdir(config_folder):
        print(f"Error: {config_folder} is not a directory.")
        sys.exit(1)

    # Find all .conf files in the directory
    search_pattern = os.path.join(config_folder, "*.conf")
    config_files = glob.glob(search_pattern)

    if not config_files:
        print(f"No .conf files found in {config_folder}")
        sys.exit(0)

    print(f"Found {len(config_files)} configuration files in {config_folder}")

    # Determine number of workers. Default to CPU count.
    # You can adjust this if you want to limit the number of parallel processes
    max_workers = os.cpu_count()
    print(f"Running simulations with {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_simulation, config_files)

