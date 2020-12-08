#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

function print_help() {
  echo 'Usage:'
  echo "  $(basename) [options] <run_script> <parameter_file>"
  echo
  echo 'Options:'
  echo '  -p   Path for job. Default: data/$(date +%Y.%m.%d)/$(basename parameter_file)'
  echo '  -n   TODO name'
  echo '  -j   Number of jobs to submit for cluster jobs. Default: max(numruns, 600)'
  echo '  -t   Estimated time per simulation, used to calculate walltime.'
  echo '       walltime = numruns / njobs * t + offset'
  echo '  -o   Offset for walltime. Default: 2. Positive only.'
  echo '  -w   Walltime for job. Overrides calculation.'
  echo '  -f   Force removal of same job name, if it exists.'
  echo '  -x   Exit job script after one run. Default off'
}

function parse_args() {
  op_path=""
  op_name=0
  op_njobs=0
  op_ncpu=1
  op_time_per=0
  op_offset=0
  op_walltime=0
  op_dry_run=1
  op_force=0
  op_exit=0
  while getopts "hp:N:j:c:t:o:w:nfx" name; do
    case $name in
      h) print_help;     exit 0;;
      p) op_path=$OPTARG;;
      N) op_name=$OPTARG;;
      j) op_njobs=$OPTARG;;
      c) op_ncpu=$OPTARG;;
      t) op_time_per=$OPTARG;;
      o) op_offset=$OPTARG;;
      w) op_walltime=$OPTARG;;
      n) op_dry_run=0;;
      f) op_force=1;;
      x) op_exit=1;;
    esac
  done
  run_script=${@:$OPTIND:1}
  parameter_file=${@:$OPTIND+1:1}
}

function check_files() {
  [ ! -f $run_script ]     && echo $run_script 'is not a file'     && exit 1
  [ ! -f $parameter_file ] && echo $parameter_file 'is not a file' && exit 1
  name=$(basename ${parameter_file%.csv})
}

function setup_path() {
  # Expects: name
  scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
  if [ -d simbiofilm ]; then
    module_path=$PWD/simbiofilm
  else
    module_path=$(python -c 'import simbiofilm as sb; print(sb.__path__[0])' 2> /dev/null)
  fi

  if [ -z $op_path ]; then
    date=$(date +%Y.%m.%d)
    if [ -z $module_path ]; then
      echo "simbiofilm module not found."
      exit 2
    fi
    ls $scriptdir/data/ > /dev/null 2>&1  # wake up the project dir, ghpcc specific
    datadir="$( cd $scriptdir/data/ && pwd -P )"
    job_path="$datadir/$date/$name"
  else
    job_path=$op_path
  fi
}

function get_walltime() {
  numruns=$(wc -l $parameter_file | awk '{print $1-2}')
  [ $op_njobs -gt 0 ] && numjobs=$op_njobs || numjobs=$(( $numruns > 600 ? 600 : $numruns ))
  [ $op_offset != '0' ] && offset=$op_offset || offset=2
  [ $op_time_per != '0' ] && time_per=$op_time_per || time_per=1
  if [ $op_walltime -gt 0 ]; then
    walltime=$op_walltime
    allowance=$((($op_walltime - $offset) * 3600))
  else
    walltime=$(python -c "print(int("$numruns"*"$time_per"/"$numjobs"+"$offset"))")
    allowance=$(python -c "print(int(("$numruns"*"$time_per"-1)*3600/"$numjobs"+"$offset"))")
  fi
  echo "Walltime: $walltime"
  echo "jobs: $numjobs"
  echo "runs: $numruns"
}

function create_files() {
  [ -d $job_path ] && [ $op_force -eq 1 ] && rm -r $job_path
  if [[ -d $job_path ]]; then
      echo "Dir exists:"
      echo "    $job_path/data"
      echo "Use -f to force remove the directory."
      exit 1
  fi

  mkdir -p $job_path/out
  mkdir -p $job_path/err
  mkdir -p $job_path/job_out
  mkdir -p $job_path/data
  cp $parameter_file $job_path/parameters.csv
  cp $run_script $job_path
  seq $numruns > $job_path/run_list.txt

  if [ $op_exit -eq 1 ]; then
    oneper='exit 0'
  else
    oneper='# exit 0 # one run'
  fi

  sed \
      -e "s|USER|$USER|" \
      -e "s|JOBNAME|$name|" \
      -e "s|WALLTIME|$walltime|" \
      -e "s|RUNCPUS|$op_ncpu|" \
      -e "s|ALLOWANCE|$allowance|" \
      -e "s|SCRIPTDIR|$scriptdir|" \
      -e "s|PYSCRIPT|$(basename $run_script)|" \
      -e "s|NUMJOBS|$numjobs|" \
      -e "s|SIMVERSION|$(cd $module_path && git rev-parse HEAD)|" \
      -e "s|FRAMEVERSION|$(cd $scriptdir && git rev-parse HEAD)|" \
      -e "s|# exit 0  # one run per job$|$oneper|" \
      -e "s|JOBPATH|$job_path|" template.job \
      > $job_path/${name}.job
  echo 1 > $job_path/${name}.count
}


parse_args $@
check_files
setup_path
get_walltime
if [ $op_dry_run -eq 1 ]; then
  create_files
  echo "Job constructed in"
  echo "    $job_path"
  # TODO: if modified, save patches:
  # git diff-index --quiet || git diff > unstaged.patch
  # git diff-index --quiet --cached HEAD -- || git diff --cached > staged.patch
fi
