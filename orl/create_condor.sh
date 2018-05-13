#!/bin/sh

# Vairables
name="convnetbasic"
model="convnetbasic"
imgdir="pixel/4/"
pwd=$(pwd)"/"
csdir="condor-scripts/"
resdir="results/pixel4/"
webpre="pixel"

# Create top level directory
mkdir -p $resdir
mkdir -p $csdir


# Normal
echo $name
rdir=$resdir$name"/"
# Create directory
mkdir -p $rdir
# Create script
script=$csdir$name".sh"
echo "#!/bin/sh" > $script
echo "export TERM=xterm-256color" >> $script
echo "export HOME=/u/richard/" >> $script
echo "export PATH=/u/richard/torch/install/bin:$PATH" >> $script
echo "export LD_LIBRARY_PATH=/u/richard/torch/install/lib:$LD_LIBRARY_PATH" >> $script
echo -n "exec th "$pwd"torch-scripts/train-on-mnist.lua" >> $script
echo -n " --train "$imgdir"train" >> $script
echo -n " --test "$imgdir"valid" >> $script
echo -n " -r 0.001" >> $script
echo -n " -b 10" >> $script
echo -n " --log" >> $script
echo -n " --rdir ../"$rdir >> $script
echo -n " --web /u/richard/public_html/ml/"$webpre >> $script
echo -n " --model "$model >> $script
chmod +x $script
# Create su
condor=$csdir$name".su"
echo "+Group = \"GRAD\"" > $condor
echo "+Project = \"OTHER\"" >> $condor
echo "+ProjectDescription = \"Breaking privacy-preserving protocols with neural networks\"" >> $condor
echo "universe = vanilla" >> $condor
echo "Executable = /bin/sh" >> $condor
echo "Requirements = InMastodon" >> $condor
echo "Notification = Always" >> $condor
echo "Notify_user = richardfmcpherson@gmail.com" >> $condor
echo "Arguments  = "$name".sh " >> $condor
echo "Error = ../"$rdir"errors" >> $condor
echo "Output = ../"$rdir"output" >> $condor
echo "Log = ../"$rdir"log" >> $condor
echo "Queue 1" >> $condor
# Submit
cd $csdir
condor_submit $name".su"
cd $pwd

