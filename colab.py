import subprocess
import time
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import time
import random
import torch

# Function to convert time from hh:mm:ss.f to seconds
def convert_time_to_seconds(time_series):
    # Convert the time strings to datetime objects
    time_format = '%H:%M:%S.%f'
    times = pd.to_datetime(time_series, format=time_format)
    # Get the first (minimum) time to use as the reference point
    min_time = times.min()
    # Subtract the minimum time and convert the time differences to seconds
    time_in_seconds = (times - min_time).dt.total_seconds()
    return time_in_seconds

def smooth_EWMA(x, alpha = 0.3):
  # Apply Exponential Weighted Moving Average smoothing
  X_EWMA = x.ewm(alpha=alpha).mean()
  return X_EWMA

def smooth_MA(x, window_size = 5):
  # Apply Moving Average smoothing
  x_MA = x.rolling(window=window_size).mean()
  return x_MA


def get_gpu_stats(device):
    gpu_power_draw=0; gpu_utilization=0; gpu_memory_utilization=0; gpu_temp=0; gpu_state='N-1'; clocks_gr=0; clocks_sm=0; clocks_mem=0
    if device =='cuda':
      try:
        # Command to get the power consumption and utilization
        command = "nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks.gr,clocks.sm,clocks.mem --format=csv,noheader,nounits"
        # Execute the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error occurred:", result.stderr)
            return gpu_power_draw, gpu_utilization, gpu_memory_utilization, gpu_temp, gpu_state, clocks_gr,clocks_sm,clocks_mem
        # Parse the output
        output = result.stdout.strip()
        stats = output.split(', ')
        gpu_power_draw = float(stats[0])  # Power consumption in watts
        gpu_utilization = int(stats[1])  # GPU utilization percentage
        gpu_memory_utilization = int(stats[2])  # memory utilization percentage
        gpu_temp = int(stats[3])  # memory utilization percentage
        gpu_state = stats[4]  # p state
        clocks_gr = int(stats[5])
        clocks_sm = int(stats[6])
        clocks_mem = int(stats[7])
      except:
        gpu_power_draw=10; gpu_utilization=0; gpu_memory_utilization=0; gpu_temp=30; gpu_state='N-2'; clocks_gr=0; clocks_sm=0; clocks_mem=0
    return gpu_power_draw, gpu_utilization, gpu_memory_utilization, gpu_temp, gpu_state, clocks_gr,clocks_sm,clocks_mem

def split_cpu_data(cpu_line):
  data = cpu_line.split(':')[-1].split(',')
  if len(data) >= 5:
    cpu_user = float(data[0].split()[0].replace('%us', ''))
    cpu_system = float(data[1].split()[0].replace('%sy', ''))
    cpu_nice = float(data[2].split()[0].replace('%ni', ''))
    cpu_idle = float(data[3].split()[0].replace('%id', ''))
    cpu_iowait = float(data[4].split()[0].replace('%wa', ''))
    cpu_irq = float(data[5].split()[0].replace('%hi', ''))
    cpu_softirq = float(data[6].split()[0].replace('%si', ''))
    cpu_usage = 100 - cpu_idle
    return cpu_user, cpu_system,cpu_nice,cpu_idle,cpu_iowait,cpu_irq,cpu_softirq,cpu_usage
  else: return None

def calc_energy(runtime,utilization, P100=61, P0=10, num_cpu=2):
  power = num_cpu*(P0 + (P100-P0)*utilization/100)
  energy = power*runtime
  #print('cpu_energy:',cpu_energy, 'Joul')
  return energy, power

def measure_resource_utilization(script_path='scr!pt_path', sampling_interval=1, output_file="utilization.csv", device = 'cuda',python_script='dum'):
    first_iter=True
    # For compatability with a previous version (to be deleted later)
    if (script_path=='scr!pt_path') and (not (python_script=='dum')):
      script_path = python_script
    #time0 = datetime.now()
    #start_time = time.time()
    # Start monitoring CPU utilization using top
    top_command = ['top', '-b', '-d', str(sampling_interval)]
    top_process = subprocess.Popen(top_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Start the Python script in the background
    if '.py' in script_path:
      process = subprocess.Popen(['python3', script_path])
    elif '.cpp' in script_path:
      subprocess.run(['g++',script_path, '-o' ,'compiled_code'])
      #process1 = subprocess.Popen(['g++',script_path, '-o' ,'compiled_code'])
      process = subprocess.Popen(['./compiled_code'])
    process_output_file = output_file.replace('.csv','_process.csv')
    # Create and open a CSV file to save CPU utilization records
    with open(output_file, mode='w', newline='') as cpu_file, open(process_output_file, mode='w', newline='') as process_file:
        cpu_writer = csv.writer(cpu_file)
        process_writer = csv.writer(process_file)
        # Execute the command
        # Write headers for CPU usage stats
        cpu_writer.writerow(["Date", "Time", "Process", "cpu_util", "memory_util","disk_util", "user", "system", "nice", "idle", "iowait", "IRQ", "softIRQ",
                              "gpu_power_draw", "gpu_util", "gpu_memory_util", "gpu_temp", "gpu_state", "clocks_gr","clocks_sm","clocks_mem"])
        # Write headers for process-specific stats
        process_writer.writerow(["Date", "Time","PID", "USER", "PR", "NI", "VIRT", "RES", "SHR", "S", "%CPU", "%MEM", "TIME+", "COMMAND"])
        total_cpu_usage = 0
        sample_count = 0
        try:
            # Monitor CPU usage while the script is running
            while process.poll() is None:  # Poll will return None while the script is running
                disk_util = psutil.disk_usage('/').percent
                cpu_line = top_process.stdout.readline()
                #print(cpu_line)
                # Capture CPU utilization line
                if "Cpu(s)" in cpu_line:
                    # Parse the line to extract various CPU statistics
                    try:
                        cpu_user, cpu_system,cpu_nice,cpu_idle,cpu_iowait,cpu_irq,cpu_softirq,cpu_usage = split_cpu_data(cpu_line)
                        total_cpu_usage += cpu_usage
                        #if first_iter:
                        #  timestamp = time0
                        #  first_iter= False
                        #else:
                        #  timestamp = datetime.now()
                        sample_count += 1
                        # Save the record in the CPU utilization CSV file
                        gpu_power_draw, gpu_utilization, gpu_memory_utilization, gpu_temp, gpu_state, clocks_gr,clocks_sm,clocks_mem = get_gpu_stats(device)
                        # Save the record in the CPU utilization CSV file
                    except:
                      print('error there')
                      #print(split_cpu_data(cpu_line))
                timestamp = datetime.now()
                if "MiB Mem" in cpu_line:
                  c = cpu_line.split()
                  memory_util = round(100*(1-(float(c[5])/float(c[3]))),1)
                  cpu_writer.writerow([timestamp.strftime('%Y-%m-%d'), timestamp.strftime('%H:%M:%S.%f'), "all", cpu_usage, memory_util,disk_util,
                                            cpu_user, cpu_system, cpu_nice, cpu_idle,
                                            cpu_iowait, cpu_irq, cpu_softirq,
                                            gpu_power_draw, gpu_utilization, gpu_memory_utilization, gpu_temp, gpu_state, clocks_gr,clocks_sm,clocks_mem])
                # Capture process-specific utilization data
                if "PID" in cpu_line:  # Process list starts right after this line
                    # Skip the next line, which is column headers
                    top_process.stdout.readline()
                    # Now, collect the data for each process
                    loop=True
                    while loop:
                        proc_line = top_process.stdout.readline().strip()
                        if proc_line:
                            # Split the line into process details
                            proc_data = proc_line.split()
                            if len(proc_data) >= 11:
                                # PID, USER, PR, NI, VIRT, RES, SHR, S, %CPU, %MEM, TIME+, COMMAND
                                process_writer.writerow([timestamp.strftime('%Y-%m-%d'), timestamp.strftime('%H:%M:%S.%f')]+proc_data[:12])
                        else:
                          loop=False
        finally:
            #print(f"Total run time: {(timestamp - time0).seconds:.0f} sec")#time.time() - start_time:.2f} sec")
            top_process.terminate()
    if sample_count > 0:
        average_cpu_utilization = total_cpu_usage / sample_count
        print(f"Average CPU Utilization: {average_cpu_utilization:.2f}%")
    else:
        print("No CPU data recorded.")


def analyze_data(csv_path="utilization.csv", with_GPU=True, smoothing_window = 1, alpha=0.2):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    time_seconds = convert_time_to_seconds(df['Time'])
    cpu_utilization = df['cpu_util']
    user_utilization = df['user']
    memory_util = df['memory_util']
    disk_util = df['disk_util']
    gpu_power_draw = df['gpu_power_draw']
    gpu_util = df['gpu_util']
    cpu_system_util = df['system']
    cpu_iowait_util = df['iowait']
    gpu_state = df['gpu_state']
    clocks_gr = df['clocks_gr']
    clocks_sm = df['clocks_sm']
    clocks_mem = df['clocks_mem']
    color_map0 = {
    'P0': 'red', 'P1': 'blue', 'P2': 'green', 'P3': 'purple', 'P4': 'orange',
    'P5': 'yellow', 'P6': 'brown', 'P7': 'pink', 'P8': 'gray', 'P9': 'cyan',
    'P10': 'magenta', 'P11': 'black', 'P12': 'olive', 'P13': 'lightblue',
    'N-1': 'lightgreen', 'N-2': 'lightcoral'}
    states_used = gpu_state.unique().tolist()
    color_map={}
    for state in states_used:
       try:
        color_map[state] = color_map0[state]
       except:
        color_map[state] = 'black'

    colors = gpu_state.map(color_map)
    # Calculate average GPU and power utilization
    print(f'Total run time: {time_seconds.max():.0f} sec')
    print(f'Average CPU Utilization: {cpu_utilization.mean():.2f}%')
    print(f'Average Memory Utilization: {memory_util.mean():.2f}%')
    print(f'Average Disk Utilization: {disk_util.mean():.2f}%')
    print(f'Average GPU Utilization: {gpu_util.mean():.2f}%')
    print(f'Average GPU Power: { gpu_power_draw.mean():.2f} W')
    runtime = time_seconds.max()
    Avg_cpu_util = cpu_utilization.mean()
    colab_cpu_energy, colab_cpu_power = calc_energy(runtime,Avg_cpu_util, P100=61, P0=10, num_cpu=2)
    print(f'Approx. Colab CPU Power: { colab_cpu_power:.0f} W')
    print(f'Approx. Colab CPU Energy: { colab_cpu_energy:.0f} J')
    
    #smoothing
    window = smoothing_window
    cpu_utilization = smooth_MA(cpu_utilization, window)
    user_utilization = smooth_MA(user_utilization, window)
    memory_util = smooth_MA(memory_util, window)
    disk_util = smooth_MA(disk_util, window)
    gpu_power_draw = smooth_MA(gpu_power_draw, window)
    gpu_util = smooth_MA(gpu_util, window)
    cpu_system_util = smooth_MA(cpu_system_util, window)
    cpu_iowait_util = smooth_MA(cpu_iowait_util, window)
    clocks_gr = smooth_MA(clocks_gr, window)
    clocks_sm = smooth_MA(clocks_sm, window)
    clocks_mem = smooth_MA(clocks_mem, window)
    Fntsize = 14
    FX = 5
    FY = 3.5
    # Create a 2x2 plot for the first four figures
    if with_GPU:
      fig, axs = plt.subplots(3, 2, figsize=(2*FX, 3*FY))
    else:
      fig, axs = plt.subplots(1, 2, figsize=(2*FX, FY))
    # Plot: CPU utilization over time
    if with_GPU:
      ax0 = axs[0, 0]
      ax1 = axs[0, 1]
    else:
      ax0 = axs[0]
      ax1 = axs[1]
    ax0.plot(time_seconds, cpu_utilization,  linestyle='--', color='k', label='total')#marker='.',
    ax0.plot(time_seconds, user_utilization,  linestyle='--', color='g', label='user')#
    ax0.plot(time_seconds, cpu_system_util, linestyle='--', color='b', label='system')#
    ax0.plot(time_seconds, cpu_iowait_util, linestyle='--', color='r', label='iowait')#
    ax0.set_xlabel('Time [s]', fontsize=Fntsize)
    ax0.set_ylabel('CPU usage %', fontsize=Fntsize)
    ax0.legend(ncol=2)#loc='upper center', bbox_to_anchor=(0.2, -0.1),
    ax0.grid(True)
    ax0.set_ylim(0,102)
    # Plot: CPU memory utilization over time
    ax1.plot(time_seconds, memory_util,linestyle='--', color='k', label='CPU Utilization')# marker='o',
    ax1.set_xlabel('Time [s]', fontsize=Fntsize)
    ax1.set_ylabel('Memory %', fontsize=Fntsize)
    ax1.grid(True)
    ax1.set_ylim(0,102)
    # Plot: disk utilization over time
    plt.figure(figsize=(FX, FY))
    plt.plot(time_seconds, disk_util,linestyle='--', color='k', label='Disk Utilization')
    plt.xlabel('Time [s]', fontsize=Fntsize)
    plt.ylabel('Disk %', fontsize=Fntsize)
    plt.grid(True)
    plt.xlim(0)
    plt.ylim(0,102)
    # Adjust layout for better display
    plt.tight_layout()
    plt.show()
    if with_GPU:
      # Plot: GPU Utilization over time
      axs[1, 0].scatter(time_seconds, gpu_util, color='blue', alpha=alpha, s=20)
      axs[1, 0].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[1, 0].set_ylabel('GPU usage %', fontsize=Fntsize)
      axs[1, 0].grid(True)
      axs[1, 0].set_ylim(0)
      # Plot: GPU Memory Utilization over time
      axs[1, 1].scatter(time_seconds, df['gpu_memory_util'], color='orange', alpha=alpha, s=20)
      axs[1, 1].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[1, 1].set_ylabel('GPU Memory %', fontsize=Fntsize)
      axs[1, 1].grid(True)
      axs[1, 1].set_ylim(0)
      # Plot: GPU Power over time
      axs[2, 0].scatter(time_seconds, gpu_power_draw, color=colors, alpha=alpha, s=20)
      axs[2, 0].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[2, 0].set_ylabel('GPU Power [W]', fontsize=Fntsize)
      axs[2, 0].grid(True)
      axs[2, 0].set_ylim(0)
      # Plot: GPU Temperature over time
      axs[2, 1].plot(time_seconds, df['gpu_temp'], color='red')
      axs[2, 1].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[2, 1].set_ylabel('GPU Temperature [C]', fontsize=Fntsize)
      axs[2, 1].grid(True)
      axs[2, 1].set_ylim(0)
      # Adjust layout for better display
      plt.tight_layout()
      plt.show()

    if with_GPU:
      fig, axs = plt.subplots(1, 2, figsize=(2*FX, 1*FY))
      '''
      # Plot: P state
      axs[0].scatter(gpu_util, gpu_power_draw, color=colors, alpha=alpha, s=20)
      axs[0].set_xlabel('GPU Usage [%]', fontsize=Fntsize)
      axs[0].set_ylabel('GPU Power [W]', fontsize=Fntsize)
      axs[0].grid(True)
      axs[0].set_xlim(0)
      axs[0].set_ylim(0,100)'''
      category_codes = {category: idx for idx, category in enumerate(color_map.keys())}
      gpu_state_code = gpu_state.map(category_codes)
      axs[0].scatter(gpu_state_code, gpu_power_draw,  color=colors, alpha=alpha, s=20)#marker='o',linestyle='-',
      #for i, color in enumerate(colors):
      #    axs[1].scatter(gpu_state_code.iloc[i], gpu_state_code.iloc[i], color=color)
      axs[0].grid(True)
      # Add category labels to the y-axis
      axs[0].set_xticks(list(category_codes.values()))
      axs[0].set_xticklabels(list(category_codes.keys()))
      axs[0].set_xlabel("Pstate", fontsize=Fntsize)
      axs[0].set_ylabel("GPU Power [W]", fontsize=Fntsize)
      axs[1].scatter(time_seconds, gpu_state_code,  color=colors, alpha=alpha, s=20)#marker='o',linestyle='-',
      for i, color in enumerate(colors):
          axs[1].scatter(time_seconds.iloc[i], gpu_state_code.iloc[i], color=color, alpha=alpha, s=20)
      axs[1].grid(True)
      # Add category labels to the y-axis
      axs[1].set_yticks(list(category_codes.values()))
      axs[1].set_yticklabels(list(category_codes.keys()))
      axs[1].set_xlabel("Time [s]", fontsize=Fntsize)
      axs[1].set_ylabel("Pstate", fontsize=Fntsize)
      plt.tight_layout()
      plt.show()
      fig, axs = plt.subplots(1, 2, figsize=(2*FX, 1*FY))
      # Plot: GPU Utilization over time
      axs[1].scatter(time_seconds, clocks_gr, color='blue', alpha=alpha, s=20)
      axs[1].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[1].set_ylabel('clocks graphics [MHz]', fontsize=Fntsize)
      axs[1].grid(True)
      #axs[1, 0].set_ylim(0)'''
      # Plot: GPU Memory Utilization over time
      '''axs[1].scatter(time_seconds, clocks_sm, color='blue', alpha=alpha, s=20)
      axs[1].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[1].set_ylabel('clocks SMs [MHz]', fontsize=Fntsize)
      axs[1].grid(True)
      #axs[1, 1].set_ylim(0)
      # Plot: GPU Power over time'''

      axs[0].scatter(time_seconds, clocks_mem, color='blue', alpha=alpha, s=20)
      axs[0].set_xlabel('Time [s]', fontsize=Fntsize)
      axs[0].set_ylabel('clocks memory [MHz]', fontsize=Fntsize)
      axs[0].grid(True)
      #axs[2, 0].set_ylim(0)
      # Adjust layout for better display
      plt.tight_layout()
      plt.show()
      '''
      # Plot: GPU Power utilization over time
      plt.figure(figsize=(FX, FY))
      plt.scatter(gpu_util, gpu_power_draw, color=colors, alpha=alpha, s=20)
      plt.xlabel('GPU Usage [%]', fontsize=Fntsize)
      plt.ylabel('GPU Power [W]', fontsize=Fntsize)
      plt.grid(True)
      plt.xlim(0)
      plt.ylim(0,100)

      # Adjust layout for better display
      plt.tight_layout()
      plt.show()'''
