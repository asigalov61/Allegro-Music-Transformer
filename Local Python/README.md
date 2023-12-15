# Allegro Installation Guide (Python Version) for Windows

This guide provides detailed steps to install the Python version of Allegro on a Windows system. Follow these steps carefully to ensure a smooth installation process.

### Prerequisites
- Windows Operating System
- Basic familiarity with command-line operations

### Installation Steps

1. **Install Chocolatey**
   - Open an administrator command prompt.
   - Copy and paste the following command:
     ```cmd
     Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
     ```
   - Wait for the installation to complete.

2. **Install FluidSynth Using Chocolatey**
   - In the same administrator command prompt, enter:
     ```cmd
     choco install fluidsynth
     ```
   - Chocolatey will automatically install FluidSynth and its dependencies.

3. **Create a Conda Environment with Python 3.10**
   - Ensure Anaconda or Miniconda is installed.
   - Create a new environment by running:
     ```cmd
     conda create -n allegro-env python=3.10
     ```
   - Activate the environment:
     ```cmd
     conda activate allegro-env
     ```

4. **Install the Latest PyTorch (Conda Version)**
   - Within the activated environment, install PyTorch by running:
     ```cmd
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
   - This command installs PyTorch with GPU support.

5. **Clone Allegro Repository**
   - Clone the Allegro repository from GitHub:
     ```cmd
     git clone https://github.com/asigalov61/Allegro-Music-Transformer
     ```
   - Move to the Local Python directory:
     ```cmd
     cd Allegro-Music-Transformer/Local Python
     ```

6. **Pip Install `requirements.txt`**
   - Ensure you have a `requirements.txt` file in your project directory.
   - Run:
     ```cmd
     pip install -r requirements.txt
     ```

7. **Place Your MIDI File in the Root Directory**
   - `piano.midi` is set as an example.

8. **Get `FluidR3_GM.sf2` SoundFont**
   - Download the sound font from [this link](https://member.keymusician.com/Member/FluidR3_GM/index.html).
   - Place the downloaded file in the project directory.

9. **Use `parameters.json` for Configuration**
   - Modify `parameters.json` in the root directory to configure Allegro as needed.

10. **Run `allegro_single_script.py`**
   - Execute the script by running:
     ```cmd
     python allegro_single_script.py
     ```
   - Ensure all dependencies are installed and configurations are set correctly.

11. **Completion**
    - Once the script finishes running, check the `content` folder for the results.
    - Your Allegro project is now set up and ready!

### Troubleshooting
- If you encounter any issues, refer to the Allegro documentation or community forums for support.

### Additional Notes
- The steps above assume a basic understanding of Python environments and Windows command-line operations.

Congratulations on setting up Allegro on your Windows system!
