#!/bin/sh

run=1

script_dir="condor_scripts/"
res_dir="results/"
path="/u/richard/p3/class/mnist/"

mkdir -p $script_dir
mkdir -p $res_dir

#TODO:
# add different types of images

for dataset in 1 10 20; do
    label="norm_"$dataset

    # Make bash script that runs torch
    script=$script_dir$label".sh"
    echo "#!/bin/sh" > $script
    echo "export TERM=xterm-256color" >> $script
    echo "export HOME=/u/richard/" >> $script
    echo "export PATH=/u/richard/torch/install/bin:\$PATH" >> $script
    echo "export LD_LIBRARY_PATH=/u/richard/torch/install/lib:\$LD_LIBRARY_PATH" >> $script
    echo -n "exec th "$path"torch-scripts/train-on-mnist.lua" >> $script 
    echo -n " --full" >> $script
    echo -n " -b 1000" >> $script 
    echo -n " --save "$res_dir$label >> $script 
    echo -n " --top 5" >> $script
    echo -n " --train "$path"mnist-data/train" >> $script
    echo -n " --test "$path"mnist-data/t10k" >> $script
    chmod +x $script

    # Make results folder
    mkdir -p $res_dir$label

    # Make and submit condor script
    condor=$script_dir$label".su"
    echo "+Group = \"GRAD\"" > $condor
    echo "+Project = \"OTHER\"" >> $condor
    echo "+ProjectDescription = \"Breaking privacy-preserving protocols with neural networks\"" >> $condor
    echo "universe = vanilla" >> $condor
    echo "Executable = /bin/sh" >> $condor
    echo "Requirements = InMastodon" >> $condor
    echo "Notification = Always" >> $condor
    echo "Notify_user = richardfmcpherson@gmail.com" >> $condor
    echo "Arguments = "$path$script >> $condor
    echo "Error = "$path$res_dir$label"/error" >> $condor
    echo "Output = "$path$res_dir$label"/output" >> $condor
    echo "Log = "$path$res_dir$label"/log" >> $condor
    echo "Queue "$run >> $condor
    condor_submit $condor
done

