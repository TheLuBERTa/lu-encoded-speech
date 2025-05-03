import os
import subprocess

class KhanomtanTTS:

  def __init__(self, version:str="1.0", device:str="cpu"):

    self.model_dir = f"./khanomtan-tts-v{version}"
    self.model_path = "best_model.pth"
    self.config_path = "config.json"
    self.language_idx = "th-th"
    self.device = device
    self.speaker_list = ['Bernard', 'Bunny', 'Caroline', 'Charel', 'Ed', 'Guy', 'Jemp', 'Judith', 'Kerstin', 'Linda', 'Luc', 'Marco', 'Nathalie', 'Sara', 'Thorsten', 'Tsyncone', 'Tsynctwo', 'p259', 'p274', 'p286'] if version == "1.0" else ['Bernard', 'Kerstin', 'Linda', 'Thorsten']

    try:
      print(f"cloning Khanomtan version {version}....")

      url = f"https://huggingface.co/wannaphong/khanomtan-tts-v{version}"

      if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
        print(f"directory for Khanomtan {version} already exists!")
      elif os.path.exists(self.model_dir) and not os.listdir(self.model_dir):
        os.remove(self.model_dir)
        self._clone_repo(url)
      else:
        self._clone_repo(url)

    except Exception as e:
      print(f"An unexpected error occurred: {e}")
  
  def _clone_repo(self, repo_url:str):

    try:

      git_lfs_install = ["git", "lfs", "install"]

      lfs_result = subprocess.run(git_lfs_install, capture_output=True, text=True, check=True)
      print("Command output:")
      print(lfs_result.stdout)
      print("Command errors (if any):")
      print(lfs_result.stderr)

      git_clone = ["git", "clone", repo_url]

      clone_result = subprocess.run(git_clone, capture_output=True, text=True, check=True)
      print("Command output:")
      print(clone_result.stdout)
      print("Command errors (if any):")
      print(clone_result.stderr)

      git_lfs_uninstall = ["git", "lfs", "uninstall"]

      lfs_result = subprocess.run(git_lfs_uninstall, capture_output=True, text=True, check=True)
      print("Command output:")
      print(lfs_result.stdout)
      print("Command errors (if any):")
      print(lfs_result.stderr)

      print("Done!")
    
    except subprocess.CalledProcessError as e:
      print(f"Error executing command: {e}")
      print("Command output:")
      print(e.stdout)
      print("Command errors:")
      print(e.stderr)
    except Exception as e:
      print(f"An unexpected error occurred: {e}")

  def __call__(self, text:str, speaker_idx:str, file_path:str, language_idx:str|None=None, device:str|None=None, verbose:bool=True):

    pre_synthesis_wd = os.getcwd()

    os.chdir(self.model_dir)

    command = [
      "tts",
      "--text", text,
      "--model_path", self.model_path,
      "--config_path", self.config_path,
      "--device", device if device else self.device,
      "--out_path", file_path,
      "--speaker_idx", speaker_idx,
      "--language_idx", language_idx if language_idx else self.language_idx
    ]

    try:
      result = subprocess.run(command, capture_output=True, text=True, check=True)
      if verbose:
        print("Command output:")
        print(result.stdout)
        print("Command errors (if any):")
        print(result.stderr)
      print(f"Successfully synthesized speech to {file_path}")

    except subprocess.CalledProcessError as e:
      print(f"Error executing command: {e}")
      print("Command output:")
      print(e.stdout)
      print("Command errors:")
      print(e.stderr)
    except FileNotFoundError:
      print("Error: The 'tts' command was not found. Make sure Coqui TTS is installed and in your PATH.")
    except Exception as e:
      print(f"An unexpected error occurred: {e}")
    
    finally:
      os.chdir(pre_synthesis_wd)
