#!/bin/sh

sudo apt install -y git openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev
sudo apt install -y python3-tk tk-dev

git clone https://github.com/yyuu/pyenv.git ~/.pyenv
export PYENV_ROOT="$HOME/.pyenv"
echo export PYENV_ROOT="\$HOME/.pyenv"  >> ~/.bashrc
export PATH="$PYENV_ROOT/bin:$PATH"
echo export PATH="\$PYENV_ROOT/bin:\$PATH"  >> ~/.bashrc
eval "$(pyenv init -)"
echo eval \"\$\(pyenv init -\)\"  >> ~/.bashrc
pyenv install 3.11.2
pyenv global 3.11.2
pip install --upgrade pip

# https://raspida.com/pip-error-pep668
mkdir ~/.pip
echo "[global]" > ~/.pip/pip.conf
echo "break-system-packages = true" >> ~/.pip/pip.conf
