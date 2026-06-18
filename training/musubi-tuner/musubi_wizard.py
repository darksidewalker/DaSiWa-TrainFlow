#!/usr/bin/env python3
import os
import subprocess
import sys

def get_input(prompt, default=None):
    if default:
        res = input(f"{prompt} [{default}]: ").strip()
        return res if res else default
    return input(f"{prompt}: ").strip()

def main():
    print("\n" + "="*60)
    print("   Musubi Tuner - Intelligent Training & Caching Wizard")
    print("="*60)
    
    # 1. Architecture Selection
    print("\n[1] Select Model Architecture:")
    print(" 1. LTX-Video 2.3 (22B)")
    print(" 2. Wan 2.2 (I2V-14B)")
    arch_choice = get_input("Choice", "1")
    
    # 2. Action Selection
    print("\n[2] Select Action:")
    print(" 1. Run Only: Pre-Cache Text Encoder Embeddings")
    print(" 2. Run Only: Pre-Cache Video Latents (VAE)")
    print(" 3. Full Training Pipeline (with Resume & Optional Caching)")
    action_choice = get_input("Choice", "3")

    is_resume = False
    resume_path = ""
    run_cache_text = False
    run_cache_latents = False

    if action_choice == "3":
        is_resume = get_input("Resume from previous state? (y/n)", "n").lower() == 'y'
        if is_resume:
            resume_path = get_input("Path to state directory (checkpoint folder)")
        run_cache_text = get_input("Run Text Encoder Caching first? (y/n)", "n").lower() == 'y'
        run_cache_latents = get_input("Run VAE Latent Caching first? (y/n)", "n").lower() == 'y'
    
    # Environment Constants
    project_root = "/home/darksidewalker/GitHub/musubi-tuner"
    python_path = f"{project_root}:{project_root}/src"
    venv_python = f"{project_root}/.venv/bin/python"
    venv_accelerate = f"{project_root}/.venv/bin/accelerate"

    # 3. Path Configuration
    print("\n[3] Path Configuration:")
    dataset_config = get_input("Path to dataset.toml", f"{project_root}/datasets/nsfw/dataset.toml")
    
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{python_path}:{existing_pp}" if existing_pp else python_path

    commands_to_run = []

    if arch_choice == "1": # LTX 2.3
        model_path = get_input("Path to LTX checkpoint", "/home/darksidewalker/GitHub/AI-Ressources/models/checkpoints/ltx-2.3-22b-dev.safetensors")
        te_path = get_input("Path to Gemma Text Encoder", "/home/darksidewalker/GitHub/AI-Ressources/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors")
        
        # Define Cache Commands
        cache_te_cmd = [
            venv_python, "ltx2_cache_text_encoder_outputs.py",
            "--dataset_config", dataset_config, "--ltx2_checkpoint", model_path,
            "--gemma_safetensors", te_path, "--ltx2_mode", "video",
            "--ltx_version", "2.3", "--mixed_precision", "bf16"
        ]
        cache_vae_cmd = [
            venv_python, "ltx2_cache_latents.py",
            "--dataset_config", dataset_config, "--ltx2_checkpoint", model_path,
            "--device", "cuda", "--vae_dtype", "bf16", "--ltx2_mode", "video"
        ]

        if action_choice == "1":
            commands_to_run.append(cache_te_cmd)
        elif action_choice == "2":
            commands_to_run.append(cache_vae_cmd)
        elif action_choice == "3":
            if run_cache_text: commands_to_run.append(cache_te_cmd)
            if run_cache_latents: commands_to_run.append(cache_vae_cmd)
            
            output_name = get_input("Output LoRA Name", "DaSiWa_LTX23_NSFW_Motion_Enhancer")
            train_cmd = [
                venv_accelerate, "launch", "--num_cpu_threads_per_process", "8", "--mixed_precision", "bf16",
                "ltx2_train_network.py", "--mixed_precision", "bf16", "--optimizer_type", "prodigyopt.Prodigy",
                "--learning_rate", "1.0", "--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=2.0", "use_bias_correction=True", "safeguard_warmup=True",
                "--lr_scheduler", "constant", "--timestep_sampling", "shifted_logit_normal",
                "--dataset_config", dataset_config, "--output_dir", "./datasets/output_ltx_video_lora",
                "--output_name", output_name, "--ltx2_checkpoint", model_path, "--gemma_safetensors", te_path,
                "--ltx_version", "2.3", "--ltx_version_check_mode", "error", "--ltx2_mode", "video",
                "--fp8_base", "--fp8_scaled", "--blocks_to_swap", "14", "--use_pinned_memory_for_block_swap",
                "--gradient_checkpointing", "--sdpa", "--network_module", "networks.lora_ltx2",
                "--network_dim", "128", "--network_alpha", "1", "--max_data_loader_n_workers", "4",
                "--persistent_data_loader_workers", "--save_every_n_epochs", "2", "--max_train_epochs", "12",
                "--save_state", "--save_state_on_train_end", "--metadata_author", "darksidewalker"
            ]
            if is_resume:
                train_cmd.extend(["--resume", resume_path])
            commands_to_run.append(train_cmd)

    elif arch_choice == "2": # Wan 2.2
        model_path = get_input("Path to Wan DiT checkpoint", "/home/darksidewalker/GitHub/AI-Ressources/models/checkpoints/wan2.2-i2v-14b-low-noise.safetensors")
        te_path = get_input("Path to T5 Encoder", "/home/darksidewalker/GitHub/AI-Ressources/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth")
        vae_path = get_input("Path to Wan VAE", "/home/darksidewalker/GitHub/AI-Ressources/models/checkpoints/wan_2.1_vae.safetensors")
        
        cache_te_cmd = [
            venv_python, "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
            "--dataset_config", dataset_config, "--t5", te_path, "--batch_size", "16"
        ]
        cache_vae_cmd = [
            venv_python, "src/musubi_tuner/wan_cache_latents.py",
            "--dataset_config", dataset_config, "--vae", vae_path, "--i2v"
        ]

        if action_choice == "1":
            commands_to_run.append(cache_te_cmd)
        elif action_choice == "2":
            commands_to_run.append(cache_vae_cmd)
        elif action_choice == "3":
            if run_cache_text: commands_to_run.append(cache_te_cmd)
            if run_cache_latents: commands_to_run.append(cache_vae_cmd)
            
            output_name = get_input("Output LoRA Name", "DaSiWa_Wan22_I2V_Physics_LoRA")
            train_cmd = [
                venv_accelerate, "launch", "--num_cpu_threads_per_process", "8", "--mixed_precision", "bf16",
                "src/musubi_tuner/wan_train_network.py", "--task", "i2v-A14B", "--mixed_precision", "bf16",
                "--optimizer_type", "prodigyopt.Prodigy", "--learning_rate", "1.0",
                "--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=2.0", "use_bias_correction=True", "safeguard_warmup=True",
                "--lr_scheduler", "constant", "--timestep_sampling", "shift", "--discrete_flow_shift", "5.0",
                "--dataset_config", dataset_config, "--output_dir", "./datasets/output_wan_video_lora",
                "--output_name", output_name, "--dit", model_path, "--t5", te_path,
                "--fp8_base", "--blocks_to_swap", "14", "--use_pinned_memory_for_block_swap",
                "--gradient_checkpointing", "--force_v2_1_time_embedding", "--sdpa",
                "--network_module", "networks.lora_wan", "--network_dim", "128", "--network_alpha", "1",
                "--max_data_loader_n_workers", "4", "--persistent_data_loader_workers",
                "--save_every_n_epochs", "2", "--max_train_epochs", "12", "--metadata_author", "darksidewalker"
            ]
            if is_resume:
                train_cmd.extend(["--resume", resume_path])
            commands_to_run.append(train_cmd)

    if not commands_to_run:
        print("\n[!] Configuration failed or no commands generated.")
        return

    print("\n" + "-"*60)
    print("PLAN TO EXECUTE:")
    for i, c in enumerate(commands_to_run):
        print(f"  Step {i+1}: {' '.join(c[:3])} ...")
    print("-"*60)
    
    confirm = get_input("Execute this plan? (y/n)", "n")
    if confirm.lower() == 'y':
        os.chdir(project_root)
        for cmd in commands_to_run:
            print(f"\n[*] Executing: {' '.join(cmd)}\n")
            try:
                subprocess.run(cmd, env=env, check=True)
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user.")
                break
            except Exception as e:
                print(f"\n[!] Error during execution: {e}")
                break

if __name__ == "__main__":
    main()