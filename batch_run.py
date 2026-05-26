import os
import sys
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_run_logs")

def run_simulation(config_path):
    try:
        # for compatibility across OS
        config_path_arg = config_path.replace(os.sep, '/')
        cmd = [sys.executable, 'run.py', config_path_arg]
        print(f"Starting simulation for {config_path}...")

        # Run the process
        result = subprocess.run(cmd, capture_output=True, text=True)

        os.makedirs(LOG_DIR, exist_ok=True)
        log_name = f"{os.path.splitext(os.path.basename(config_path))[0]}.log"
        log_path = os.path.join(LOG_DIR, log_name)

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"Config: {config_path}\n")
            log_file.write(f"Return code: {result.returncode}\n\n")
            log_file.write("=== STDOUT ===\n")
            log_file.write(result.stdout or "")
            if result.stdout and not result.stdout.endswith("\n"):
                log_file.write("\n")
            log_file.write("\n=== STDERR ===\n")
            log_file.write(result.stderr or "")
            if result.stderr and not result.stderr.endswith("\n"):
                log_file.write("\n")

        if result.returncode == 0:
            return True, config_path, result.stdout, log_path
        else:
            return False, config_path, f"Error running {config_path}:\n{result.stderr}\nOutput:\n{result.stdout}", log_path

    except Exception as e:
        return False, config_path, f"Exception while running {config_path}: {e}", None

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

    total = len(config_files)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_simulation, config) for config in config_files]

        for future in as_completed(futures):
            success, path, output, log_path = future.result()
            completed += 1
            remaining = total - completed

            if success:
                print(f"Successfully finished {path} ({remaining} left).\n"
                      f"    Log: {log_path}")
            else:
                print(output)
                if log_path:
                    print(f"Log: {log_path}")
