#!/bin/bash

setup_git() {
  # openssl aes-256-cbc -K $encrypted_9279eeca0411_key -iv $encrypted_9279eeca0411_iv -in github_gsl_for_gsl_sync.enc -out ~/.ssh/id_rsa -d
  # chmod 600 ~/.ssh/id_rsa
  # echo -e "Host github.com\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
  ssh-add -l
  ssh-add ~/.ssh/id_rsa_9f905b139576d663cb8a69d3b7ce34d5
  git config --global user.email "circleci@circleci.com"
  git config --global user.name "CircleCI"
}

setup_repository() {
  if [[ ! -e ~/.cache/gsl ]]; then
    git clone git@github.com:i05nagai/gsl.git ~/.cache/gsl
  fi
  cd ~/.cache/gsl
  git remote add gnu git://git.savannah.gnu.org/gsl.git
}

sync() {
  git checkout master
  git pull gnu master -r
  git push origin master
}

setup_git
setup_repository
sync
