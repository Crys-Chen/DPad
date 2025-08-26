# Copyright 2025 Xinhua Chen, Duke CEI Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import shlex  
from params import parser

if __name__ == "__main__":
    args=parser.parse_args()
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

    # --- Directory creation logic (unchanged) ---
    for model in ['instruct', '1.5', 'base']:
        if not os.path.exists(f"output/log/{model}/"):
            os.makedirs(f"output/log/{model}/")
        if not os.path.exists(f"output/debug/{model}/"):
            os.makedirs(f"output/debug/{model}/")

    # --- Model selection logic (unchanged) ---
    if args.model == 'instruct':
        model = 'GSAI-ML/LLaDA-8B-Instruct'
    elif args.model == 'base':
        model = 'GSAI-ML/LLaDA-8B-Base'
    elif args.model == '1.5':
        model = 'GSAI-ML/LLaDA-1.5'
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Parameter and filename construction logic (unchanged) ---
    if args.threshold == 0:
        steps = args.gen_length
        sampling = '_t_'
        threshold = ''
    else:
        assert 0 < args.threshold < 1, "Invalid threshold"
        steps = args.gen_length // args.block_size
        sampling = '_p_'
        threshold = f'threshold={args.threshold},'

    if args.dc is True:
        args.c = True
        cache = 'dc_'
    elif args.c is True:
        cache = 'c_'
    else:
        cache = ''

    if args.e is True:
        early = 'e_'
    else:
        early = ''
 
    if args.name != '':
        args.name += '_'

    if args.d == 'gaussian':
        dropout = f"_g_sigma{args.k_sigma}_scale{args.scale}_win{args.window}"
    elif args.d == 'random':
        dropout = f"_r_tk{args.nt}_win{args.window}"
    else:
        assert args.d == 'null', "Invalid dropout strategy"
        dropout = ''

    filename = f"{args.name}{args.task}{sampling}{cache}{early}len{args.gen_length}_blk{args.block_size}{dropout}"
    log_file = f"output/log/{args.model}/{filename}.log"
    debug_file = f"output/debug/{args.model}/{filename}.log"
    save_dir = f"output/checkpoint/{args.model}/{filename}"

    # --- [The Elegant Way] Build and execute the command using subprocess ---

    # 1. Build the command as a list of arguments for safety and clarity.
    base_cmd = ['accelerate', 'launch', 'eval_llada.py']

    task_args = ['--tasks', args.task]
    if args.task == 'humaneval':
        task_args.append('--log_samples')
    else:
        task_args.extend(['--num_fewshot', str(args.num_fewshot)])

    model_args_string = (
        f"model_path={model},gen_length={args.gen_length},steps={steps},block_length={args.block_size},"
        f"{threshold}"
        f"from_scratch={args.re},save_dir={save_dir},show_speed=True,"
        f"use_cache={args.c},dual_cache={args.dc},early_termination={args.e},"
        f"dropout={args.d},sigma={args.k_sigma},scale={args.scale},preserved_tokens={args.nt},window={args.window}"
    )

    # 2. Assemble the final command list.
    cmd_list = base_cmd + task_args
    cmd_list.extend(['--confirm_run_unsafe_code', '--model', 'llada_dist'])
    cmd_list.extend(['--model_args', model_args_string])

    # Add the specific output path parameter based on the task.
    if args.task == 'humaneval':
        output_path = f"output/humaneval_results/{model}/{filename}"
        cmd_list.extend(['--output_path', output_path])

    # 3. Print the command 
    print(shlex.join(cmd_list))
    print("-" * 50)  # Separator

    # 4. Execute the command
    try:
        # Use a 'with' statement to open log files, ensuring they are properly closed afterward.
        with open(log_file, 'w') as log_f, open(debug_file, 'w') as debug_f:
            # Execute the command, redirecting stdout and stderr to the log files.
            print(f"   Log file: {log_file}")
            print(f"   Debug file: {debug_file}")
            subprocess.run(
                cmd_list,
                stdout=log_f,       # Redirect standard output
                stderr=debug_f,       # Redirect standard error
                check=True          # Raise an exception on non-zero exit codes (errors)
            )
        print(f"\n✅ Command completed successfully.")

    except FileNotFoundError:
        print(f"❌ Error")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code: {e.returncode}.")
        print(f"   Check the debug log for details: {debug_file}")