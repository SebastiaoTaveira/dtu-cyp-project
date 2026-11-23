module load python3/3.12.16
source .DL/bin/activate
unset SSH_ASKPASS GIT_ASKPASS
echo "Ready: $(which python)"
run () {
  python finetune.py "$@" && python log_results.py
}

